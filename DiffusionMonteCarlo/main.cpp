/*
Diffusion Monte Carlo Code
Implemented by Christian Cosgrove, 2016

Based on theory & pseudocode from Paul Kent, 
http://web.ornl.gov/~pk7/thesis/pkthnode21.html
*/

#include <glm\glm.hpp>
#include<iostream>
#include<fstream>
#include <vector>
#include <random>
#include <glm\gtx\norm.hpp>
#include <chrono>
#include "Walker.h"
#include <string>

//Number of electrons
#define N 1

using namespace glm;
using namespace std;

mt19937 randgen(chrono::high_resolution_clock::now().time_since_epoch().count());

vector<Walker<N>> walkers;
const int N_walkers_initial = 2000;


//IMPORTANT: adjust with potential
//Ensure that the trial wavefunction contains all of the atoms
const double alpha = 1.3; // stddev of trial wave function

const double dt = 0.01;

const int TCOUNT = 1e5;
const int BLOCK_SIZE = 500;
const int EQUILIBRATION_TIME = 80;//in blocks

const double CREF = 0.003;

const double PI = 3.14159265359;


const double ENERGY_CLAMP = -300.0;

double bondLength = 1.39;

//Refers to the embedding potential
enum class SystemType {
	HydrogenAtom,
	DihydrogenCation,
	HeliumAtom,
	LithiumAtom,
	BerylliumAtom
};

const SystemType CurrSystemType = SystemType::HydrogenAtom;


const string filename = "li.csv";

//sample from gaussian trial wave function
dvec3 sample_trial_function() {
	normal_distribution<double> dist(0, alpha);
	return dvec3(dist(randgen), dist(randgen), dist(randgen));
}

//n samples
array<dvec3,N> sample_trial_function_n() {
	array<dvec3, N> out;
	
	for (int i = 0; i < N; i++)
	{
		/*if (i == 0) out[i] = glm::dvec3(-0.9, 0, 0);
		else if (i == 1) out[i] = glm::dvec3(0.9, 0, 0);*/
		out[i] = sample_trial_function();
	}
	return out;
}

double gen_norm(double mean = 0, double stddev = 1) {
	normal_distribution<double> dist(mean, stddev);
	return dist(randgen);
}
double gen_uniform(double min = 0.0, double max = 1.0) {
	uniform_real_distribution<double> dist(min, max);
	return dist(randgen);
}
int gen_uniform(int min = 0.0, int max = 1.0) {
	uniform_int_distribution<int> dist(min, max);
	return dist(randgen);
}


//unnormalized trial wavefunction (centered at 0)
double trial_function(dvec3 position) {
	double dist2 = glm::length2(position);
	return exp(-dist2 / (2 * alpha*alpha));
}

double local_energy(dvec3 position) {
	double dist2 = glm::length2(position);
	double tfu = trial_function(position);
	double kinetic = (-dist2 + 3 * alpha*alpha) / (2 * alpha * alpha * alpha * alpha);
	double potential;
	switch (CurrSystemType)
	{
	case SystemType::HydrogenAtom:
		potential = -1.0 / sqrt(dist2);
		break;
	case SystemType::DihydrogenCation:
		potential = -1.0 / length(position - glm::dvec3(-bondLength / 2, 0, 0)) - 1.0 / length(position - glm::dvec3(bondLength / 2, 0, 0));
		break;
	case SystemType::HeliumAtom:
		potential = -2.0 / sqrt(dist2);
		break;
	case SystemType::LithiumAtom:
		potential = -3.0 / sqrt(dist2);
		break;
	case SystemType::BerylliumAtom:
		potential = -4.0 / sqrt(dist2);
		break;
	}

	return kinetic + potential;
}

double walker_energy(const Walker<N>& walker) {
	double energy = 0;
	for (int i = 0; i < N; i++)
	{
		energy += local_energy(walker.positions[i]);
		for (int j = 0; j < i; j++) //don't double count
		{
			//electron-electron repulsion
			energy += 1.0 / glm::length(walker.positions[i] - walker.positions[j]);
		}
	}
	return std::max(energy,ENERGY_CLAMP); //because of the pathology of the Coulomb potential, clamp energy to prevent instability
}


dvec3 quantum_force(dvec3 position) {
	return -position / (alpha*alpha);
}

double greens_function_ratio(dvec3 newpos, dvec3 pos) {
	double arg = 0;
	arg += glm::length2(newpos - pos - 0.5 * dt * quantum_force(pos));
	arg -= glm::length2(pos - newpos - 0.5 * dt * quantum_force(newpos));
	arg /= (2 * dt);
	return exp(arg);
}

double greens_function(dvec3 newpos, dvec3 pos, double trial_energy) {
	double gb = exp((-0.5 *(local_energy(newpos) + local_energy(pos)) - trial_energy)*dt);
	double gd = pow(2 * PI * dt, 3.0 / 2) * exp(-glm::length2(pos - newpos - 0.5*dt * quantum_force(newpos)) / (2 * dt));
	return gb*gd;
}

double branching_factor(const Walker<N>& newpos, const Walker<N>& pos, double trial_energy) {
	double localavg = walker_energy(newpos) + walker_energy(pos);
	localavg /= 2;
	return exp(-(dt * (localavg - trial_energy)));
}

double hartreeToEV(double energy) { return energy * 27.21139; }

void duplicateWalker(vector<Walker<N>>& walkers, int index)
{
	if (!walkers[index].alive) throw std::logic_error("Cannot duplicate a dead walker!");
	for (int i = 0; i < walkers.size(); i++) {
		if (!walkers[i].alive) {
			walkers[i].positions[0] = walkers[index].positions[0];
			walkers[i].alive = true;
			return;
		}
	}
	//failed to duplicate walker (saturated)
}

int numWalkers(vector<Walker<N>>& walkers) {
	int count = 0;
	for (Walker<N>& w : walkers) if (w.alive) count++;
	return count;
}

//thin out walkers
void renormalize(vector<Walker<N>>& walkers) {
	/*while (walkers.size() > N_walkers_initial)
	walkers.erase(walkers.begin() + gen_uniform(0, walkers.size() - 1));
	int s = walkers.size();
	while (walkers.size() < N_walkers_initial)
	{
	if (s == 0) walkers.push_back(sample_trial_function());
	else walkers.push_back(walkers[gen_uniform(0,s-1)]);
	}*/

	while (numWalkers(walkers) > N_walkers_initial)
	{
		walkers[gen_uniform(0, walkers.size() - 1)].alive = false;
	}
	while (numWalkers(walkers) < N_walkers_initial)
	{
		walkers[gen_uniform(0, walkers.size() - 1)].alive = true;
	}

	/*for (int i = 0; i < walkers.size(); i++)
	{
	if (!walkers[i].alive) {
	int randInd = -1;
	if (randInd == -1 || randInd == i) randInd = gen_uniform(0, walkers.size() - 1);
	walkers[i] = walkers[randInd];
	}
	}*/
}


void initFile()
{
	fstream str(filename, ios::out | ios::app);
	str << "bond length,energy" << endl;
}

//returns ground state energy
double calculate() 
{
	//for (bondLength = 3; bondLength < 5.5; bondLength += 0.05)
	//{
		double trial_energy = 0;

		double accEnergy = 0;
		double energyCount = 0;
		//initialize walkers
		walkers.clear();
		walkers.reserve(N_walkers_initial);
		for (int i = 0; i < N_walkers_initial; i++)
		{
			auto pos = sample_trial_function_n();
			walkers.push_back(Walker<N>(pos));
		}

		for (Walker<N>& w : walkers)
			trial_energy += walker_energy(w);
		trial_energy /= walkers.size();
		for (int i = 0; i < N_walkers_initial; i++) {
			auto pos = sample_trial_function_n();
			Walker<N> w = (Walker<N>(pos));
			w.alive = false;
			walkers.push_back(w);

		}//add excess walkers

		for (int t = 0; t < TCOUNT / BLOCK_SIZE; t++)
		{
			if (t < EQUILIBRATION_TIME)
			{
				accEnergy = 0;
				energyCount = 0;
			}
			for (int bi = 0; bi < BLOCK_SIZE; bi++)
			{
				for (int w = 0; w < walkers.size(); w++)
				{
					if (!walkers[w].alive) continue;

					Walker<N> prevpos = walkers[w];
					for (int e = 0; e < N; e++)
					{
						dvec3 pos = walkers[w].positions[e];
						double dt12 = sqrt(dt);
						dvec3 eta = dvec3(gen_norm(0, dt12), gen_norm(0, dt12), gen_norm(0, dt12));

						//update walker
						dvec3 newpos = pos + eta + dt * quantum_force(pos);

						double weight = trial_function(newpos) / trial_function(pos);
						weight *= weight;
						weight *= greens_function(newpos, pos, trial_energy) / greens_function(pos, newpos, trial_energy);

						//Metrpolis step
						if (gen_uniform(0.0, 1.0) < weight)
						{
							pos = newpos;
							walkers[w].positions[e] = pos;
						}
					}
					//calculate branching factor
					double branching = branching_factor(walkers[w], prevpos, trial_energy); // acts as a weighting

					//if (t > EQUILIBRATION_PERIOD) {
					{
						accEnergy += branching*walker_energy(walkers[w]);
						energyCount += branching;
					}
					double copyref = branching + gen_uniform(0.0, 1.0);
					int copies = std::min<int>(int(copyref), 3); // number of copies; if 0, delete walker

					if (copies == 0) {
						walkers[w].alive = false;
					}
					else if (copies > 1) {
						for (int i = 0; i < copies - 1; i++)
							duplicateWalker(walkers, w);
					}
				}
			}

			//if (t < EQUILIBRATION_PERIOD)
		{
			//if (energyCount!=0)
			//trial_energy = (accEnergy / energyCount);
			//trial_energy = accEnergy / energyCount;
			int n = numWalkers(walkers);
			//trial_energy = 1.5;
			trial_energy -= CREF / dt * log((double)numWalkers(walkers) / N_walkers_initial);
		}/*}
		 else {
		 if (energyCount!=0)
		 trial_energy += (accEnergy / energyCount - trial_energy) * 0.01;
		 }*/
			renormalize(walkers);
			cout << "E = " << accEnergy / energyCount << "\tE_R = " << trial_energy << "\tn = " << numWalkers(walkers) << "\tblen = " << bondLength << "\tdone: " << 100 * t * BLOCK_SIZE / TCOUNT <<"%"<< endl;

		}
		{
			fstream str(filename, ios::out | ios::app);
			str << bondLength << "," << accEnergy / energyCount << endl;
		}
		return accEnergy / energyCount;
	//}
}


int main()
{
	initFile();
	calculate();
}