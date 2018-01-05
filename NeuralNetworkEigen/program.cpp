#include <iostream>
#include <Eigen\Dense>
#include <random>
#include "NeuralNetwork.h"
#include "DifferentialEvolution.h"
#include "Structures.h"
#include <vector>

using Eigen::MatrixXd;
using namespace std;

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

vector<StockData> ReadStockData()
{
	FILE *stream;
	vector<StockData> r;
	StockData sd;

	if (fopen_s(&stream, "C:\\Coding\\snp.txt", "r") == 0) {
		while (fscanf_s(stream, "%zu,%lf,%lf,%lf,%lf,%zu", &sd.Date, &sd.Open, &sd.High, &sd.Low, &sd.Close, &sd.Volume) == 5) {
			sd.Volume = 0;
			r.push_back(sd);
		}

		while (fscanf_s(stream, "%zu,%lf,%lf,%lf,%lf,%zu", &sd.Date, &sd.Open, &sd.High, &sd.Low, &sd.Close, &sd.Volume) == 6) {
			r.push_back(sd);
		}

		fclose(stream);
	}

	delete stream;

	return r;
}

vector<StockData> FilterMinimumDate(vector<StockData> &data, size_t minimumDate)
{
	vector<StockData> r;

	for (size_t i = 0; i < data.size(); i++)
	{
		if (data[i].Date >= minimumDate)
		{
			r.push_back(data[i]);
		}
	}

	return r;
}

double BuyBars(vector<StockData> &data, int *indices, int startIndex)
{
	double r = 0.0;
	double stocksHold = 0.0;
	double counter = 1.0;

	for (size_t i = startIndex; i < data.size(); i++)
	{
		if (indices[i])
		{
			r += data[i].Open * counter;
			stocksHold += counter;
			counter = 1.0;
		}
		else
		{
			counter += 1.0;
		}
	}

	if (counter > 1.0)
	{
		counter -= 1.0;
		r += data[data.size() - 1].Open * counter;
		stocksHold += counter;
	}

	return r;
}

const int POPULATION = 42;

void StockStuff()
{
	//20 features, 10 inputs, beide tanh => 5350982 bei epoch 12948
	//30 features, 5 inputs, beide relu => 5380929 bei epoch 13305
	//30 features, 5 inputs, beide tanh => 5323048 bei epoch 3247
	int samples;
	int features = 30;
	int inputNeurons = 5;
	int outputs = 1;

	size_t sampleIndex;
	vector<StockData> data;
	vector<StockData> sinceDate;
	std::default_random_engine re;
	std::uniform_int_distribution<int> zeroOrOneDistribution(0, 1);
	std::uniform_int_distribution<int> populationDistribution(0, POPULATION - 1);
	int *p;
	double allBars, base, stocksHold, allBarsRise, changedRise;
	MatrixXd X;

	//idee: man kann doch auch das optimum suchen lassen, mhhh aber dann ohne regel

	data = ReadStockData();
	sinceDate = FilterMinimumDate(data, 20000000);
	samples = sinceDate.size() - features - 1;
	p = new int[samples];
	X = MatrixXd::Ones(samples, features + 1); //bias ist das plus 1


	for (size_t i = features + 1; i < sinceDate.size(); i++)
	{
		base = sinceDate[i - 1].Open; //das soll ja für den nächsten tag also i gelten
		sampleIndex = i - features - 1;

		for (size_t n = 0; n < features; n++)
		{
			X(sampleIndex, n) = (base / sinceDate[sampleIndex + n].Open) - 1.0;
		}
	}

	//average mit std von den errors berechenen, wenn std sehr klein dann pulsen

	//cout << X << std::endl;
	for (size_t i = 0; i < samples; i++)
	{
		p[i] = 1;
	}

	allBars = BuyBars(sinceDate, p, features + 1);
	stocksHold = (double)samples;
	allBarsRise = stocksHold * sinceDate[sinceDate.size() - 1].Open / allBars;


	LinearInputBiasLayer input(samples, features, inputNeurons);
	TanhLayer inputActivation;
	LinearBiasLayer hidden(samples, inputNeurons, outputs);
	TanhLayer activation;



	MatrixXd inputWeights[POPULATION];
	MatrixXd hiddenWeights[POPULATION];

	MatrixXd inputChallenger(features + 1, inputNeurons);
	MatrixXd inputMutator(features + 1, inputNeurons);

	MatrixXd hiddenChallenger(inputNeurons + 1, outputs);
	MatrixXd hiddenMutator(inputNeurons + 1, outputs);


	Eigen::MatrixXi is;
	double f = 0.5;
	double eMutator, eMin;
	double errors[POPULATION];
	int alphaIndex, omegaIndex, epoch;
	eMin = std::numeric_limits<double>::max();
	epoch = 0;


	//hier gehts los
	input.X = X;

	for (size_t i = 0; i < POPULATION; i++)
	{
		inputWeights[i] = MatrixXd::Random(features + 1, inputNeurons);
		hiddenWeights[i] = MatrixXd::Random(inputNeurons + 1, outputs);

		input.W = inputWeights[i];
		input.Forward();
		inputActivation.Forward(input.Y);

		hidden.W = hiddenWeights[i];
		hidden.Forward(inputActivation.Y);
		activation.Forward(hidden.Y);

		is = activation.Y.unaryExpr(&zeroOrOne).cast<int>();
		p = is.data();
		errors[i] = BuyBars(sinceDate, p, features + 1);
	}

	do
	{
		for (size_t i = 0; i < POPULATION; i++)
		{
			do
			{
				alphaIndex = populationDistribution(re);
				omegaIndex = populationDistribution(re);
			} while (alphaIndex == i || omegaIndex == i || omegaIndex == alphaIndex); //do I care? => yes much faster

			CalculateChallenger(inputChallenger, inputWeights[i], inputWeights[alphaIndex], inputWeights[omegaIndex], f);
			CalculateMutator(inputMutator, inputChallenger, inputWeights[i]);


			CalculateChallenger(hiddenChallenger, hiddenWeights[i], hiddenWeights[alphaIndex], hiddenWeights[omegaIndex], f);
			CalculateMutator(hiddenMutator, hiddenChallenger, hiddenWeights[i]);


			input.W = inputMutator; //reicht challenger vll schon aus? => nö ist doof
			input.Forward();
			inputActivation.Forward(input.Y);

			hidden.W = hiddenMutator;
			hidden.Forward(inputActivation.Y);
			activation.Forward(hidden.Y);

			is = activation.Y.unaryExpr(&zeroOrOne).cast<int>();
			p = is.data();
			eMutator = BuyBars(sinceDate, p, features + 1);

			if (eMutator < errors[i])
			{
				inputWeights[i] = inputMutator;
				hiddenWeights[i] = hiddenMutator;
				errors[i] = eMutator;

				if (eMutator < eMin)
				{
					eMin = eMutator;

					changedRise = (stocksHold * sinceDate[sinceDate.size() - 1].Open / eMin) - allBarsRise;
					printf("%i: %.0f -> %.0f     %.5f\n", epoch, allBars, eMin, changedRise);
				}
			}
		}

		epoch++;

	} while (true);

	//n tage vorher rausfischen, ergebnis ist 0 oder 1, bei 1 kaufen, bei 0 weiterlaufen lassen
	//die summe der investitionen soll möglichst klein sein
	//wenn nicht gekauft wird, wird beim nächsten kauf die anzahl der nichtkäufe dazugenommen
	//2 tage nicht kaufen, 3. tag kaufen => 3 mal zu diesem tagespreis kaufen

	//train => besser geworden upward pull auf alle?!
	//schlechter geworden negative pull auf die einser
	//egal erstmal differential stuff
}


void Challenger()
{
	int samples = 20; //immer dran denken, es kann auch sein, dass nicht gefittet werden kann! (nahe 0 vll unmöglich) das netzwerk ist vll garnicht in der lage
	int features = 5;
	int outputs = 2;

	LinearInputBiasLayer input(samples, features, outputs);
	TanhLayer activation;
	MatrixXd targets(samples, outputs);

	MatrixXd weights[POPULATION];
	MatrixXd challenger(features + 1, outputs);
	MatrixXd mutator(features + 1, outputs);
	MatrixXd alpha;
	MatrixXd omega;
	double f = 0.5;
	double e, eMutator, eMin;
	int alphaIndex, omegaIndex, epoch;

	eMin = std::numeric_limits<double>::max();
	std::default_random_engine re;
	std::uniform_int_distribution<int> dist(0, POPULATION - 1);
	epoch = 0;

	for (size_t i = 0; i < samples; i++)
	{
		targets(i, 0) = -1.0;
		targets(i, 1) = 1.0;
	}

	for (size_t i = 0; i < POPULATION; i++)
	{
		weights[i] = MatrixXd::Random(features + 1, outputs);
	}

	do
	{
		for (size_t i = 0; i < POPULATION; i++)
		{
			input.W = weights[i];
			input.Forward();
			activation.Forward(input.Y);
			e = activation.CalculateError(targets);

			do
			{
				alphaIndex = dist(re);
				omegaIndex = dist(re);
			} while (alphaIndex == i || omegaIndex == i || omegaIndex == alphaIndex); //do I care? => yes much faster

			CalculateChallenger(challenger, weights[i], weights[alphaIndex], weights[omegaIndex], f);
			CalculateMutator(mutator, challenger, weights[i]);

			input.W = mutator; //reicht challenger vll schon aus? => nö ist doof
			input.Forward();
			activation.Forward(input.Y);
			eMutator = activation.CalculateError(targets);
			//hier kann ich die fitness methode einbauen fitness(activation.Y)

			if (eMutator < e)
			{
				weights[i] = mutator;

				if (eMutator < eMin)
				{
					eMin = eMutator;
				}
			}
		}

		epoch++;

		if (epoch % 10 == 0)
		{
			printf("%i: %f\n", epoch, eMin);

		}
	} while (true);
}


void ChallengerNN()
{
	int samples = 20;
	int features = 5;
	int outputs = 2;

	LinearInputBiasLayer input(samples, features, outputs);
	TanhLayer activation;
	MatrixXd topGradients;
	MatrixXd targets(samples, outputs);
	double e, eMin;
	int epoch;

	eMin = std::numeric_limits<double>::max();
	epoch = 0;

	for (size_t i = 0; i < samples; i++)
	{
		targets(i, 0) = -1.0;
		targets(i, 1) = 1.0;
	}

	do
	{
		input.Forward();
		activation.Forward(input.Y);
		e = activation.CalculateError(targets);


		topGradients = activation.Y - targets;

		activation.Backward(topGradients);
		input.Backward(activation.dYwrtX);
		input.Update(0.0001);

		if (e < eMin)
		{
			eMin = e;
		}

		epoch++;

		if (epoch % 1 == 0)
		{
			printf("%i: %f\n", epoch, eMin);

		}
	} while (true);
}

int main()
{

	//Challenger();


	StockStuff();
	int selfSupported1 = InvestUntilSelfSupportedQuarterly(1200, 50, 1.0, 1.06, 1.03);
	int selfSupported2 = InvestUntilSelfSupportedQuarterlyDividendsAsStockPricePart(1000, 50, 1.0, 1.04, 0.04);
	Savings();
	Layer();

	return 0;
}