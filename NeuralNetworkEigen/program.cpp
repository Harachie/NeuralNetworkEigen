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
		dYwrtX = topGradients.cwiseProduct(Y.unaryExpr(&tanhDerivative));
	}

	double CalculateError(MatrixXd &targets)
	{
		return (Y - targets).cwiseProduct(Y - targets).sum() * 0.5;
	}

};

struct LinearInputLayer
{
	int Samples;
	int Features;
	int Outputs;

	MatrixXd X;
	MatrixXd W;
	MatrixXd Y;
	MatrixXd dYwrtW;

	LinearInputLayer(int samples, int features, int outputs)
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
		dYwrtW = X.transpose() * topGradients;
	}

	void Update(double learningRate)
	{
		W = W - dYwrtW * learningRate;
	}


};

struct LinearInputBiasLayer
{
	int Samples;
	int Features;
	int Outputs;

	MatrixXd X;
	MatrixXd W;
	MatrixXd Y;
	MatrixXd dYwrtW;

	LinearInputBiasLayer(int samples, int features, int outputs)
	{
		Samples = samples;
		Features = features;
		Outputs = outputs;

		X = MatrixXd(samples, features + 1);
		W = MatrixXd(features + 1, outputs);

		X.setRandom();
		W.setRandom();
	}

	void Forward()
	{
		Y = X * W;
	}

	void Backward(MatrixXd &topGradients)
	{
		dYwrtW = X.transpose() * topGradients;
	}

	void Update(double stepSize) //input layer can only update their W
	{
		W = W - dYwrtW * stepSize;
	}


};

struct LinearLayer
{
	int Samples;
	int Features;
	int Outputs;

	MatrixXd InternalX;
	MatrixXd W;
	MatrixXd Y;
	MatrixXd dYwrtX;
	MatrixXd dYwrtW;

	LinearLayer(int samples, int features, int outputs)
	{
		Samples = samples;
		Features = features;
		Outputs = outputs;

		W = MatrixXd(features, outputs);
		W.setRandom();
	}

	void Forward(MatrixXd &X)
	{
		InternalX = X;
		Y = X * W;
	}

	void Backward(MatrixXd &topGradients)
	{
		dYwrtX = topGradients * W.transpose(); //die gehen an den unteren layer weiter
		dYwrtW = InternalX.transpose() * topGradients;
	}

	void Update(double stepSize)
	{
		W = W - dYwrtW * stepSize;
	}


};

struct LinearBiasLayer
{
	int Samples;
	int Features;
	int Outputs;

	MatrixXd InternalX;
	MatrixXd W;
	MatrixXd Y;
	MatrixXd dYwrtX;
	MatrixXd dYwrtW;

	LinearBiasLayer(int samples, int features, int outputs)
	{
		Samples = samples;
		Features = features;
		Outputs = outputs;

		InternalX = MatrixXd(samples, features + 1);
		W = MatrixXd(features + 1, outputs);
		W.setRandom();
	}

	void Forward(MatrixXd &X)
	{
		InternalX << X, MatrixXd(X.rows(), 1).setOnes();
		Y = InternalX * W;
	}

	void Backward(MatrixXd &topGradients)
	{
		dYwrtX = topGradients * W.transpose().leftCols(Features); //die gehen an den unteren layer weiter
		dYwrtW = InternalX.transpose() * topGradients;
	}

	void Update(double stepSize)
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
	int samples = 4;
	int features = 2;
	int inputNeurons = 3;
	int hidden1Neurons = 2;
	int outputNeurons = 2;
	double e;
	double learningRate = 0.01;
	size_t i = 0;

	LinearInputBiasLayer input(samples, features, inputNeurons);
	TanhLayer tanhInputActivation;
	LinearBiasLayer hiddenLayer(samples, inputNeurons, hidden1Neurons);
	TanhLayer tanhHidden1Activation;
	LinearBiasLayer outputLayer(samples, hidden1Neurons, outputNeurons);
	TanhLayer tanhOutputActivation;

	MatrixXd targets(samples, hidden1Neurons);
	MatrixXd topGradients;

	for (size_t i = 0; i < samples; i++)
	{
		targets(i, 0) = -1.0;
		targets(i, 1) = 1.0;
	}

	
	do
	{
		input.Forward();
		tanhInputActivation.Forward(input.Y);
		hiddenLayer.Forward(tanhInputActivation.Y);
		tanhHidden1Activation.Forward(hiddenLayer.Y);
		outputLayer.Forward(tanhHidden1Activation.Y);
		tanhOutputActivation.Forward(outputLayer.Y);


		topGradients = tanhOutputActivation.Y - targets;

		tanhOutputActivation.Backward(topGradients);
		outputLayer.Backward(tanhOutputActivation.dYwrtX);

		tanhHidden1Activation.Backward(outputLayer.dYwrtX);
		hiddenLayer.Backward(tanhHidden1Activation.dYwrtX);

		tanhInputActivation.Backward(hiddenLayer.dYwrtX);
		input.Backward(tanhInputActivation.dYwrtX);

		outputLayer.Update(learningRate);
		hiddenLayer.Update(learningRate);
		input.Update(learningRate);

		e = tanhOutputActivation.CalculateError(targets);

		if (i++ % 10000 == 0)
		{
			std::cout << e << std::endl;
		}
	} while (e > 0.001);

	std::cout << "Error: " << e << std::endl;
	std::cout << "Epoch: " << i << std::endl;
	std::cout << tanhOutputActivation.Y << std::endl;

}

int main()
{
	Layer();

	return 0;
}