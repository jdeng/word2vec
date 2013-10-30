#include <vector>

typedef std::vector<float> Vector;

namespace v {
	inline float dot(const Vector&x, const Vector& y) { 
		int m = x.size(); const float *xd = x.data(), *yd = y.data();
		float sum = 0.0;
		while (--m >= 0) sum += (*xd++) * (*yd++);
		return sum;
	}

	inline void saxpy(Vector& x, float g, const Vector& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) (*xd++) += g * (*yd++);
	}

	inline void unit(Vector& x) {
		float len = ::sqrt(dot(x, x));
		if (len == 0) return;

		int m = x.size(); float *xd = x.data();
		while (--m >= 0) (*xd++) /= len;
	}
}

