#include <iostream>
#include <Eigen\Dense>
#include <random>

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
	int samples = 40000;
	int features = 13;
	int inputNeurons = 7;
	int hidden1Neurons = 5;
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

	MatrixXd targets(samples, outputNeurons);
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
	} while (1); //e > 0.00001

	std::cout << "Error: " << e << std::endl;
	std::cout << "Epoch: " << i << std::endl;
	//std::cout << tanhOutputActivation.Y << std::endl;

}

double *InterestRatesConstant(int years, double interest)
{
	double *r = new double[years];

	for (size_t i = 0; i < years; i++)
	{
		r[i] = interest;
	}

	return r;
}

void InterestRatesRandom(double *r, int years, double interest, std::uniform_real_distribution<double> &dist, std::default_random_engine &re)
{
	double multiplied = 1.0;
	double expected = pow(interest, years);
	double factor;
	double itemFactor;

	for (size_t i = 0; i < years; i++)
	{
		r[i] = dist(re);
		multiplied *= r[i];
	}

	factor = expected / multiplied;
	itemFactor = pow(factor, 1.0 / years);

	for (size_t i = 0; i < years; i++)
	{
		r[i] *= itemFactor;
	}
}

double Savings(double startCapital, double yearlyInvest, int years, double *interestRates, double dividendRate)
{
	double money;
	double dividends;
	double dividendsAfterTax;
	double tax = 0.0;

	money = startCapital;

	for (size_t i = 0; i < years; i++)
	{
		money += yearlyInvest;
		dividends = money * dividendRate;
		dividendsAfterTax = dividends * 0.7375;
		money -= dividends;
		money += dividendsAfterTax;
		tax += dividends - dividendsAfterTax;
		money *= interestRates[i];
	}

	return money;
}

double Savings(double startCapital, double yearlyInvest, int years, double *interestRates)
{
	double money;

	money = startCapital;

	for (size_t i = 0; i < years; i++)
	{
		money += yearlyInvest;
		money *= interestRates[i];
	}

	return money;
}

double Savings(double startCapital, double yearlyInvest, int years, double interest)
{
	double money;

	money = startCapital;

	for (size_t i = 0; i < years; i++)
	{
		money += yearlyInvest;
		money *= interest;
	}

	return money;
}

double Invest(double startCapital, double yearlyInvest, int years)
{
	double money;

	money = startCapital;

	for (size_t i = 0; i < years; i++)
	{
		money += yearlyInvest;
	}

	return money;
}

void Savings()
{
	int years = 30;
	int epochs = 50000;
	int higherThanBase = 0;
	int lowerThanBase = 0;
	int lowerThanInvested = 0;
	double interestRate = 1.07;
	double startCapital = 0.0;
	double monthlyInvestment = 1200;
	double yearlyInvestment = 12 * monthlyInvestment;
	double mean;
	double money;
	double base;
	double invested;
	double *interestRates = new double[years];
	double minimumMoney = std::numeric_limits<double>::max();
	double maximumMoney = std::numeric_limits<double>::min();
	std::default_random_engine re;

	base = Savings(startCapital, yearlyInvestment, years, InterestRatesConstant(years, interestRate), 0.015);
	invested = Invest(startCapital, yearlyInvestment, years);

	for (double minRate = 0.00; minRate < 1.0; minRate += 0.01)
	{
		minimumMoney = std::numeric_limits<double>::max();
		maximumMoney = std::numeric_limits<double>::min();

		higherThanBase = 0;
		lowerThanBase = 0;
		lowerThanInvested = 0;
		mean = 0.0;
		std::uniform_real_distribution<double> dist(interestRate - minRate, interestRate + minRate);


		for (size_t i = 0; i < epochs; i++)
		{
			InterestRatesRandom(interestRates, years, interestRate, dist, re);
			money = Savings(startCapital, yearlyInvestment, years, interestRates, 0.015);

			higherThanBase += (money > base);
			lowerThanBase += (money < base);
			lowerThanInvested += (money < invested);
			mean += money;

			if (money > maximumMoney)
			{
				maximumMoney = money;
			}

			if (money < minimumMoney)
			{
				minimumMoney = money;
			}
		}

		printf("%f: %.0f - %.0f - %.0f | %i - %i | %i\n", minRate, minimumMoney, (mean / (double)epochs), maximumMoney, lowerThanBase, higherThanBase, lowerThanInvested);
	}

}

int InvestUntilSelfSupportedQuarterly(double perMonth, double startCommodityCost, double startDividends, double interestRate, double dividendRate)
{
	int i = 0;
	double dividends;
	double dividendsAfterTax = 0.0;
	double invested = 0.0;
	double stocksHold = 0.0;
	double dividendsPerStock = startDividends;
	double costsPerStock = startCommodityCost;

	do
	{
		i++;
		invested += perMonth;
		stocksHold += (perMonth / costsPerStock);

		if (i % 4 == 0)
		{
			dividends = stocksHold * dividendsPerStock * 0.25;
			costsPerStock -= (dividendsPerStock * 0.25); //das kost ja nun weniger, weil die dividenden abgezogen werden
			dividendsAfterTax = dividends * 0.7375;

			invested += dividendsAfterTax;
			stocksHold += (dividendsAfterTax / costsPerStock);
		}

		if (i % 12 == 0) //könnte auch auf pro monat runterbrechen... so ist es etwas positiver ;)
		{
			dividendsPerStock *= dividendRate;
			costsPerStock *= interestRate;
		}
	} while (dividendsAfterTax / 4 < perMonth);

	return i;
}

int InvestUntilSelfSupportedQuarterlyDividendsAsStockPricePart(
	double perMonth, double startCommodityCost, double startDividends,
	double interestRate, double dividendRate)
{
	int i = 0;
	double dividends;
	double dividendsAfterTax = 0.0;
	double invested = 0.0;
	double stocksHold = 0;
	double costsPerStock = startCommodityCost;
	double dividendsPerStock;

	do
	{
		i++;
		invested += perMonth;
		stocksHold += (perMonth / costsPerStock);

		if (i % 4 == 0)
		{
			dividendsPerStock = costsPerStock * dividendRate * 0.25;
			dividends = stocksHold * dividendsPerStock;
			costsPerStock -= dividendsPerStock; //das kost ja nun weniger, weil die dividenden abgezogen werden

			if (dividends > 200.0)
			{
				dividendsAfterTax = 200.0 + (dividends - 200.0) * 0.7375;
			}
			else
			{
				dividendsAfterTax = dividends; //keine steuer
			}

			invested += dividendsAfterTax;
			stocksHold += (dividendsAfterTax / costsPerStock);
		}

		if (i % 12 == 0) //könnte auch auf pro monat runterbrechen... so ist es etwas positiver ;)
		{
			costsPerStock *= interestRate;
		}
	} while (dividendsAfterTax / 4 < perMonth);

	return i;
}

int main()
{
	int selfSupported1 = InvestUntilSelfSupportedQuarterly(1200, 50, 1.0, 1.06, 1.03);
	int selfSupported2 = InvestUntilSelfSupportedQuarterlyDividendsAsStockPricePart(1000, 50, 1.0, 1.04, 0.04);
	Savings();
	Layer();

	return 0;
}