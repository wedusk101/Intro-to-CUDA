#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <cmath>
#include <iostream> 
#include <string>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cuda.h>

void errorCheck(cudaError_t code, const char *func, const char *fileName, const int line)
{
	if (code)
	{
		std::cerr << "CUDA error = " << (int)code << " in file: " <<
			fileName << " function: " << func << " on line: " << line << "\n";
		cudaDeviceReset();
		exit(1);
	}
}

#define cudaErrorCheck(arg) errorCheck( (arg), #arg, __FILE__, __LINE__ )

struct Vec3
{
	float x;
	float y;
	float z;

	__host__ __device__ Vec3() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3(const float &x_, const float &y_, const float &z_) : x(x_), y(y_), z(z_) {}

	__host__ __device__ float getMagnitude() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ Vec3 getNormalized() const
	{
		float mag = getMagnitude();
		return Vec3(x / mag, y / mag, z / mag);
	}

	__host__ __device__ Vec3 operator+(const Vec3 &v) const // addition
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}

	__host__ __device__ Vec3 operator-(const Vec3 &v) const // subtraction
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}

	__host__ __device__ Vec3 operator*(const float &c) const // scalar multiplication
	{
		return Vec3(c * x, c * y, c * z);
	}

	__host__ __device__ Vec3 operator/(const float &c) const // scalar division
	{
		return Vec3(x / c, y / c, z / c);
	}

	__host__ __device__ float operator%(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}

	__host__ __device__ Vec3 operator&(const Vec3 &v) const // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	__host__ __device__ float dot(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}
};

struct Ray
{
	Vec3 o; // origin
	Vec3 d; // direction
	mutable float t;
	float tMin;
	mutable float tMax;

	__host__ __device__ Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_), t(INT_MAX), tMin(0.1), tMax(INT_MAX) {}
};

struct Geometry
{
	Vec3 color;

	__host__ __device__ virtual bool intersects(const Ray &ray) const = 0;
	__host__ __device__ virtual Vec3 getNormal(const Vec3 &point) const = 0;
};

struct Sphere : public Geometry
{
	Vec3 center;
	float radius;

	__host__ __device__ Sphere(const Vec3 &c, const float &rad, const Vec3 &col) : center(c), radius(rad)
	{
		color = col;
	}

	__host__ __device__ Vec3 getNormal(const Vec3 &point) const // returns the surface normal at a point
	{
		return (point - center) / radius;
	}

	__host__ __device__ bool intersects(const Ray &ray) const
	{
		const float eps = 1e-4;
		const Vec3 oc = ray.o - center;
		const float b = 2 * (ray.d % oc);
		const float a = ray.d % ray.d;
		const float c = (oc % oc) - (radius * radius);
		float delta = b * b - 4 * a * c;
		if (delta < eps) // discriminant is less than zero
			return false;
		delta = sqrt(delta);
		const float t0 = (-b + delta) / (2 * a);
		const float t1 = (-b - delta) / (2 * a);
		ray.t = (t0 < t1) ? t0 : t1;
		if (ray.t >= ray.tMin && ray.t <= ray.tMax)
		{
			ray.tMax = ray.t;
			return true;
		}
		else
			return false;
	}
};

struct Plane : public Geometry
{
	Vec3 normal; // normal of the plane
	Vec3 point; // a point on the plane


	__host__ __device__ Plane(const Vec3 &n, const Vec3 &p, const Vec3 &col) : normal(n), point(p)
	{
		color = col;
	}

	__host__ __device__ Vec3 getNormal(const Vec3 &point) const
	{
		return normal;
	}

	__host__ __device__ bool intersects(const Ray &ray) const
	{
		const double eps = 1e-4;
		double parameter = ray.d % normal;
		if (fabs(parameter) < eps) // ray is parallel to the plane
			return false;
		ray.t = ((point - ray.o) % normal) / parameter;
		if (ray.t >= ray.tMin && ray.t <= ray.tMax)
		{
			ray.tMax = ray.t;
			return true;
		}
		else
			return false;
	}
};

struct Light
{
	Vec3 position;
	float radius;
	Vec3 color;
	float intensity;

	__host__ __device__ Light(const Vec3 &position_, const float &radius_, const Vec3 &color_, const float &intensity_) : position(position_), radius(radius_), color(color_), intensity(intensity_) {}
};

struct Camera
{
	Vec3 position;
	Vec3 direction;

	// add a lower left corner for orientation

	__host__ __device__ Camera(const Vec3 &pos, const Vec3 &dir) : position(pos), direction(dir) {}
};

__device__ Vec3 colorModulate(const Vec3 &lightColor, const Vec3 &objectColor) // performs component wise multiplication for colors  
{
	return Vec3(lightColor.x * objectColor.x, lightColor.y * objectColor.y, lightColor.z * objectColor.z);
}
__device__ void clamp(Vec3 &col)
{
	col.x = (col.x > 1) ? 1 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 1) ? 1 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 1) ? 1 : (col.z < 0) ? 0 : col.z;
}

__device__ Vec3 getPixelColor(Ray &cameraRay, Geometry **scene, int sceneSize, const Light *light)
{
	Vec3 pixelColor;
	Vec3 white(1, 1, 1);
	bool hitStatus = false;
	int hitIndex = 0;
	for (int i = 0; i < sceneSize; ++i)
	{
		if (scene[i]->intersects(cameraRay))
		{
			hitStatus = true;		
			hitIndex = i;
		}
	}

	if (hitStatus)
	{
		Vec3 surf = cameraRay.o + cameraRay.d * cameraRay.tMax; // point of intersection
		Vec3 L = (light->position - surf).getNormalized();


		// check for shadows
		Ray shadowRay(surf, L);
		for (int i = 0; i < sceneSize; ++i)
			if (scene[i]->intersects(shadowRay))
				return pixelColor;

		Vec3 N = scene[hitIndex]->getNormal(surf).getNormalized();
		float diffuse = L.dot(N);
		pixelColor = (colorModulate(light->color, scene[hitIndex]->color) + white * diffuse) * light->intensity;
		clamp(pixelColor);
	}
	return pixelColor;
}

__global__ void render(Vec3 *fb, int width, int height, const Camera *camera, Geometry **scene, int sceneSize, const Light *light)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((x >= width) || (y >= height))
		return;
	int index = y * width + x;
	Ray cameraRay(Vec3(x, y, 0), camera->direction); // camera ray from each pixel 
	fb[index] = getPixelColor(cameraRay, scene, sceneSize, light);
}

__global__ void initScene(int width, int height, Camera *camera, Geometry **scene, Light *light)
{
	Vec3 white(1, 1, 1);
	Vec3 blue(0, 0, 1);
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// light = new Light(Vec3(0.8 * width, 0.25 * height, 100), 1, white, 0.5);
		// camera = new Camera(Vec3(0.5 * width, 0.5 * height, 0), Vec3(0, 0, 1));

		light->position = Vec3(0.8 * width, 0.25 * height, 100);
		light->radius = 1;
		light->color = Vec3(1, 1, 1);
		light->intensity = 0.5;

		camera->position = Vec3(0.5 * width, 0.5 * height, 0);
		camera->direction = Vec3(0, 0, 1);

		scene[0] = new Sphere(Vec3(0.5 * width, 0.45 * height, 1000), 100, Vec3(1, 0, 0));
		scene[1] = new Sphere(Vec3(0.65 * width, 0.2 * height, 600), 50, Vec3(0, 0, 1));
		scene[2] = new Plane(Vec3(0, 0, -1), Vec3(0.5 * width, 0.5 * height, 1500), Vec3(1, 1, 0));
		scene[3] = new Sphere(Vec3(0.5 * width, 0.52 * height, 700), 35, Vec3(0, 1, 1));
	}
}

int main()
{
	int width = 2560;
	int height = 1440;
	int tx = 8;
	int ty = 8;

	// colors (R, G, B)
	const Vec3 white(1, 1, 1);
	const Vec3 black(0, 0, 0);
	const Vec3 red(1, 0, 0);
	const Vec3 green(0, 1, 0);
	const Vec3 blue(0, 0, 1);
	const Vec3 cyan(0, 1, 1);
	const Vec3 magenta(1, 0, 1);
	const Vec3 yellow(1, 1, 0);

	Light *light; 
	Camera *camera;
	Geometry **scene;
	int sceneSize = 4;

	int numPixels = width * height;
	size_t fbSize = numPixels * sizeof(Vec3);

	Vec3 *fb;
	cudaErrorCheck(cudaMallocManaged((void**)&fb, fbSize));
	cudaErrorCheck(cudaMallocManaged((void**)&light, sizeof(Light)));
	cudaErrorCheck(cudaMallocManaged((void**)&camera, sizeof(Camera)));
	cudaErrorCheck(cudaMallocManaged((void**)&scene, sizeof(Geometry*)));
	
	dim3 threadsPerBlock(tx, ty);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	auto start = std::chrono::high_resolution_clock::now();
	initScene << <1, 1>> > (width, height, camera, scene, light);
	render << <numBlocks, threadsPerBlock >> > (fb, width, height, camera, scene, sceneSize, light);
	cudaErrorCheck(cudaGetLastError());
	cudaErrorCheck(cudaDeviceSynchronize());
	auto stop = std::chrono::high_resolution_clock::now();

	std::ofstream out("result.ppm"); // creates a PPM image file for saving the rendered output
	out << "P3\n" << width << " " << height << "\n255\n";

	for (int i = 0; i < numPixels; ++i)
		out << (int)(255.99 * fb[i].x) << " " << (int)(255.99 * fb[i].y) << " " << (int)(255.99 * fb[i].z) << "\n"; // write out the pixel values

	std::cout << "\nTime taken was " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)).count() << " milliseconds." << std::endl;
	cudaErrorCheck(cudaFree(fb));
	cudaErrorCheck(cudaFree(light));
	cudaErrorCheck(cudaFree(camera));
	cudaErrorCheck(cudaFree(scene));
}