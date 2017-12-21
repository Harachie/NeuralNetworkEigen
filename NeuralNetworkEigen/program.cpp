#include <iostream>
#include <Eigen\Dense>

using Eigen::MatrixXd;

double customTanh(double x)
{
	return std::tanh(x);
}

double tanhDerivative(double x)
{
	double s;

	s = std::tanh(x);

	return 1.0 - s * s;
}

struct AddGate
{
	MatrixXd F;
	MatrixXd X;
	MatrixXd Y;
	MatrixXd dFwrtXLocal;
	MatrixXd dFwrtYLocal;
	MatrixXd dFwrtX;
	MatrixXd dFwrtY;

	AddGate(int rows, int columns)
	{
		X = MatrixXd(rows, columns);
		Y = MatrixXd(rows, columns);
		dFwrtXLocal = MatrixXd(rows, columns).setOnes();
		dFwrtYLocal = MatrixXd(rows, columns).setOnes();
	}

	void Forward()
	{
		F = X + Y;
	}

	void Backward(MatrixXd &topGradients)
	{
		dFwrtX = dFwrtXLocal.cwiseProduct(topGradients);
		dFwrtY = dFwrtYLocal.cwiseProduct(topGradients);
	}

	void UpdateX(double stepSize)
	{
		X = X + dFwrtX * stepSize;
	}

	void UpdateY(double stepSize)
	{
		Y = Y + dFwrtY * stepSize;
	}

};

struct MultiplyGate
{
	MatrixXd F;
	MatrixXd X;
	MatrixXd Y;
	MatrixXd dFwrtXLocal;
	MatrixXd dFwrtYLocal;
	MatrixXd dFwrtX;
	MatrixXd dFwrtY;

	MultiplyGate(int rows, int columns)
	{
		X = MatrixXd(rows, columns);
		Y = MatrixXd(rows, columns);
		dFwrtXLocal = MatrixXd(rows, columns);
		dFwrtYLocal = MatrixXd(rows, columns);
	}

	void Forward()
	{
		F = X.cwiseProduct(Y);
		dFwrtXLocal = Y;
		dFwrtYLocal = X;
	}

	void Backward(MatrixXd &topGradients)
	{
		dFwrtX = dFwrtXLocal.cwiseProduct(topGradients);
		dFwrtY = dFwrtYLocal.cwiseProduct(topGradients);
	}

	void UpdateX(double stepSize)
	{
		X = X + dFwrtX * stepSize;
	}

	void UpdateY(double stepSize)
	{
		Y = Y + dFwrtY * stepSize;
	}


};

struct TanhLayer
{
	MatrixXd Y;
	MatrixXd dYwrtX;

	void Forward(MatrixXd &X)
	{
		Y = X.unaryExpr(&customTanh);
	}

	void Backward(MatrixXd &topGradients)
	{
		dYwrtX = topGradients.cwiseProduct(Y.unaryExpr(&tanhDerivative)); //das ist wohl falsch... net von Z sonders von X sollte es sein (vll?) ne passt
	}

	double CalculateError(MatrixXd &targets)
	{
		return (Y - targets).cwiseProduct(Y - targets).sum() * 0.5;
	}

};

struct LinearLayer
{
	int Samples;
	int Features;
	int Outputs;

	MatrixXd Y;
	MatrixXd X;
	MatrixXd W;
	MatrixXd dYwrtX;
	MatrixXd dYwrtW;

	LinearLayer(int samples, int features, int outputs)
	{
		Samples = samples;
		Features = features;
		Outputs = outputs;

		X = MatrixXd(samples, features);
		W = MatrixXd(features, outputs);

		X.setRandom();
		W.setRandom();
	}

	void Forward()
	{
		Y = X * W;
	}

	void Backward(MatrixXd &topGradients)
	{
		dYwrtX = topGradients * W.transpose();
		dYwrtW = X.transpose() * topGradients;
	}

	void UpdateX(double stepSize)
	{
		X = X - dYwrtX * stepSize;
	}

	void UpdateW(double stepSize)
	{
		W = W - dYwrtW * stepSize;
	}


};


void Gates()
{
	AddGate add(1, 1);
	MultiplyGate mul(1, 1);
	MatrixXd topGradients(1, 1);
	double stepSize = 0.01;

	add.X(0) = -2;
	add.Y(0) = 5;
	mul.Y(0) = -4;
	topGradients(0) = 1;

	add.Forward();
	mul.X = add.F;
	mul.Forward();

	mul.Backward(topGradients);
	add.Backward(mul.dFwrtX);

	mul.UpdateY(stepSize);
	add.UpdateX(stepSize);
	add.UpdateY(stepSize);
}

void Layer()
{
	int samples = 10000;
	int features = 2;
	int hidden = 3;
	int outputs = 2;
	double e;
	double stepSize = 0.00001;
	size_t i = 0;

	LinearLayer input(samples, features, hidden);
	TanhLayer tanFirstActivation;
	LinearLayer hiddenLayer(samples, hidden, outputs);
	TanhLayer tanSecondActivation;

	MatrixXd targets(samples, outputs);
	MatrixXd topGradients;

	for (size_t i = 0; i < samples; i++)
	{
		targets(i, 0) = -1.0;
		targets(i, 1) = 1.0;
	}

	
	do
	{
		input.Forward();
		tanFirstActivation.Forward(input.Y);
		hiddenLayer.X = tanFirstActivation.Y;
		hiddenLayer.Forward();
		tanSecondActivation.Forward(hiddenLayer.Y);



		topGradients = tanSecondActivation.Y - targets;

		tanSecondActivation.Backward(topGradients);
		hiddenLayer.Backward(tanSecondActivation.dYwrtX);
		tanFirstActivation.Backward(hiddenLayer.dYwrtX);
		input.Backward(tanFirstActivation.dYwrtX);

		hiddenLayer.UpdateW(stepSize);
		input.UpdateW(stepSize);

		if (i++ % 10 == 0)
		{
			e = tanSecondActivation.CalculateError(targets);
			std::cout << e << std::endl;
		}
	} while (true);

}

int main()
{
	Layer();

	return 0;
}