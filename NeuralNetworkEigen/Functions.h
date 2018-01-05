#pragma once
#include <math.h>

inline double oneMinusX(double x)
{
	return 1.0 - x;
}

inline double zeroOrOne(double x)
{
	return x > 0;
}

inline double customRelu(double x)
{
	return (x > 0) * x;
}

inline double customTanh(double x)
{
	return tanh(x);
}

inline double tanhDerivative(double x)
{
	double s;

	s = tanh(x);

	return 1.0 - s * s;
}