#pragma once
#include "Functions.h"
#include <Eigen\Dense>

using Eigen::MatrixXd;

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

struct NeuralNetworkTanh
{

	NeuralNetworkTanh()
	{

	}

};