#pragma once
#include <Eigen\Dense>

using Eigen::MatrixXd;

// der nimmt einfach nen random von 0-1 und behält den jeweiligen anteil an der source, und den rest (1 - [0-1]) vom challenger
// bei 0.7 nimmt er 70% vom source und 1-0.7 = 30% vom anderen
void CalculateMutator(MatrixXd &mutator, MatrixXd &challenger, MatrixXd &source)
{
	MatrixXd crossover = MatrixXd::Random(source.rows(), source.cols()).cwiseAbs(); //anstatt zero or none (unaryExpr(&zeroOrOne)) kann man auch einfach abs nehmen... dann ists das wie oben beschreiben (prozentuale anteile)

	mutator << source.cwiseProduct(crossover) + challenger.cwiseProduct(crossover.unaryExpr(&oneMinusX));
}

void CalculateChallenger(MatrixXd &challenger, MatrixXd &source, MatrixXd &alpha, MatrixXd &omega, double f)
{
	challenger << source + f * (alpha - omega);
}

MatrixXd CreateChallenger(MatrixXd &source, MatrixXd &alpha, MatrixXd &omega, double f)
{
	MatrixXd challenger(source.rows(), source.cols());

	CalculateChallenger(challenger, source, alpha, omega, f);

	return challenger;
}
