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

void errorCheck(cudaError_t code, const char *fileName, const int line)
{
	if (code)
	{
		std::cerr << "CUDA error = " << (int)code << " in " <<
			fileName << " on line: " << line << "\n";
		cudaDeviceReset();
		exit(1);
	}
}


#define cudaErrorCheck(arg) errorCheck( (arg), __FILE__, __LINE__ )

struct Vec3
{
	double x;
	double y;
	double z;

	Vec3() : x(0), y(0), z(0) {}
	Vec3(const double &x_, const double &y_, const double &z_) : x(x_), y(y_), z(z_) {}

	double getMagnitude() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	Vec3 getNormalized() const
	{
		double mag = getMagnitude();
		return Vec3(x / mag, y / mag, z / mag);
	}

	Vec3 operator+(const Vec3 &v) const // addition
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}

	Vec3 operator-(const Vec3 &v) const // subtraction
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}

	Vec3 operator*(const double &c) const // scalar multiplication
	{
		return Vec3(c * x, c * y, c * z);
	}

	Vec3 operator/(const double &c) const // scalar division
	{
		return Vec3(x / c, y / c, z / c);
	}

	double operator%(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}

	Vec3 operator&(const Vec3 &v) const // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	double dot(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}
};

struct Ray
{
	Vec3 o; // origin
	Vec3 d; // direction

	Vec3 closestHitPoint;

	Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_), closestHitPoint(Vec3(INT_MAX, INT_MAX, INT_MAX)) {}
};

struct Sphere
{
	Vec3 center;
	double radius;
	Vec3 color;

	bool hasBeenHit;
	double distanceToCamera;

	Sphere(const Vec3 &c, const double &rad, const Vec3 &col) : center(c), radius(rad), color(col), hasBeenHit(false), distanceToCamera(0) {}

	Vec3 getNormal(const Vec3 &point) const // returns the surface normal at a point
	{
		return (point - center) / radius;
	}

	bool intersects(const Ray &ray, double &t) const
	{
		const double eps = 1e-4;
		const Vec3 oc = ray.o - center;
		const double b = 2 * (ray.d % oc);
		const double a = ray.d % ray.d;
		const double c = (oc % oc) - (radius * radius);
		double delta = b * b - 4 * a * c;
		if (delta < eps) // discriminant is less than zero
			return false;
		delta = sqrt(delta);
		const double t0 = (-b + delta) / (2 * a);
		const double t1 = (-b - delta) / (2 * a);
		t = (t0 < t1) ? t0 : t1;
		return true;
	}
};

struct Plane
{
	Vec3 normal; // normal of the plane
	Vec3 point; // a point on the plane
	Vec3 color;

	bool hasBeenHit;
	double distanceToCamera;

	Plane(const Vec3 &n, const Vec3 &p, const Vec3 &c) : normal(n), point(p), color(c), hasBeenHit(false), distanceToCamera(0) {}

	Vec3 getNormal() const
	{
		return normal;
	}

	bool intersects(const Ray &ray, double &t) const
	{
		const double eps = 1e-4;;
		double parameter = ray.d % normal;
		if (fabs(parameter) < eps) // ray is parallel to the plane
			return false;
		t = ((point - ray.o) % normal) / parameter;
		return true;
	}
};

struct Light
{
	Vec3 position;
	double radius;
	Vec3 color;
	double intensity;

	Light(const Vec3 &position_, const double &radius_, const Vec3 &color_, const double &intensity_) : position(position_), radius(radius_), color(color_), intensity(intensity_) {}
};

struct Camera
{
	Vec3 position;
	Vec3 direction;

	// add a lower left corner for orientation

	Camera(const Vec3 &pos, const Vec3 &dir) : position(pos), direction(dir) {}
};

inline double dot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double getEuclideanDistance(const Vec3 &a, const Vec3 &b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
	return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

Vec3 colorModulate(const Vec3 &lightColor, const Vec3 &objectColor) // performs component wise multiplication for colors  
{
	return Vec3((lightColor.x / 255) * objectColor.x, (lightColor.y / 255) * objectColor.y, (lightColor.z / 255) * objectColor.z);
}

void clamp(Vec3 &col)
{
	col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main()
{
	// setup resolution, camera, colors, objects and lights

	const int height = 480;
	const int width = 640;


	// colors (R, G, B)
	const Vec3 white(255, 255, 255);
	const Vec3 black(0, 0, 0);
	const Vec3 red(255, 0, 0);
	const Vec3 green(0, 255, 0);
	const Vec3 blue(0, 0, 255);
	const Vec3 cyan(0, 255, 255);
	const Vec3 magenta(255, 0, 255);
	const Vec3 yellow(255, 255, 0);

	const Camera camera(Vec3(0.5 * width, 0.5 * height, 0), Vec3(0, 0, 1)); // scene camera

	// scene objects and lights
	Sphere sphere(Vec3(0.5 * width, 0.45 * height, 350), 25, blue); // blue sphere
	Plane plane(Vec3(0, 0, -1), Vec3(0.5 * width, 0.5 * height, 500), cyan); // yellow plane

	Light light(Vec3(0.8 * width, 0.25 * height, 100), 1, white, 0.5); // white scene light
	const Vec3 ambient(128, 0, 0);	// light red ambient light
	const double ambientIntensity = 0.25;

	Vec3 pixelColor(0, 0, 0);	// set background color to black 

	double t = 0;

	auto start = std::chrono::high_resolution_clock::now();

	std::ofstream out("result.ppm"); // creates a PPM image file for saving the rendered output
	out << "P3\n" << width << " " << height << "\n255\n";

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			pixelColor = ambient * ambientIntensity; // default color of each pixel
			Ray cameraRay(Vec3(x, y, 0), camera.direction); // camera ray from each pixel 

			sphere.hasBeenHit = false; // used for determining whether a ray has already intersected the sphere before intersecting the plane
			sphere.distanceToCamera = INT_MAX;

			plane.hasBeenHit = false;
			plane.distanceToCamera = INT_MAX;

			if (sphere.intersects(cameraRay, t))
			{
				Vec3 surf = cameraRay.o + cameraRay.d * t; // point of intersection
				Vec3 L = (light.position - surf).getNormalized();
				Vec3 N = sphere.getNormal(surf).getNormalized();
				sphere.distanceToCamera = getEuclideanDistance(cameraRay.o, surf);
				sphere.hasBeenHit = true;
				if (sphere.distanceToCamera < getEuclideanDistance(cameraRay.o, cameraRay.closestHitPoint))
					cameraRay.closestHitPoint = surf;
				Ray shadowRay(surf, L); // shadow ray from point of intersection in the direction of the light source
				double diffuse = dot(L, N);
				pixelColor = (colorModulate(light.color, sphere.color) + white * diffuse) * light.intensity + ambient * ambientIntensity; // white * diffuse = highlight 
				clamp(pixelColor);
			}

			if (plane.intersects(cameraRay, t))
			{
				Vec3 surf = cameraRay.o + cameraRay.d * t;
				Vec3 L = (light.position - surf).getNormalized();
				Vec3 N = plane.getNormal().getNormalized();
				plane.distanceToCamera = getEuclideanDistance(cameraRay.o, surf);
				plane.hasBeenHit = true;
				if (plane.distanceToCamera < getEuclideanDistance(cameraRay.o, cameraRay.closestHitPoint))
					cameraRay.closestHitPoint = surf;
				Ray shadowRay(surf, L);
				double diffuse = dot(L, N);
				if (!sphere.hasBeenHit && !sphere.intersects(shadowRay, t) || plane.distanceToCamera < sphere.distanceToCamera && !sphere.intersects(shadowRay, t)) // hacky
					pixelColor = (colorModulate(light.color, plane.color) + white * diffuse) * light.intensity + ambient * ambientIntensity;
				clamp(pixelColor);
			}

			out << (int)pixelColor.x << " " << (int)pixelColor.y << " " << (int)pixelColor.z << "\n"; // write out the pixel values
		}
	}

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "\nTime taken was " << (std::chrono::duration_cast<std::chrono::seconds>(stop - start)).count() << " seconds." << std::endl;
}