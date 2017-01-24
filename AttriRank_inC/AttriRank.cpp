/* =========================================================================================================
 
 C++ Implementation of AttriRank
   Author: Ming-Han Feng
 
 for more details, please refer to the paper:
   Unsupervised Ranking using Graph Structures and Node Attributes
   Chin-Chi Hsu, Yi-An Lai, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin
   Web Search and Data Mining (WSDM), 2017
 
 === Requirements: g++ ===
  compile: g++ -std=c++11 -O2 AttriRank.cpp -o AttriRank
    usage: AttriRank EdgeFile AttriFile [options]
  
     << options >>
    --unweighted      (none)                  graph is unweighted (default: weighted)
    --undirected      (none)                  graph is undirected (default: directed)
    -k, --kernel    [rbf_ap|rbf|cosine]       kernel used in AttriRank (default: rbf_ap)
    -i, --iter      [MaximumIterations]       maximum number of iterations in power method (default: 100)
    -c, --conv      [ConvergenceThreshold]    the convergence threshold in power method (default: 1.0e-6)
    -d, --damp      [start,step,end]          damping factor (default: 0.0,0.2,1.0)
    -t, --total     [alpha,beta]              TotalRank parameters (default: 1,1)
  
  e.g. AttriRank graph.edge graph.attri -d 0.7,0.02,0.9 -t 1e-9 --unweighted
  e.g. AttriRank edge.txt attri.txt --undirected -i 200 -k rbf
 
 === EdgeFile format ===
  EachLine: NodeFromID<integer> NodeToID<integer> (weight<float/integer>)
  Note: the weight is set to 1.0 in weighted version if there is no third value provided in a line
  
  e.g. 0 1
       2 3
       2 4
  e.g. 1 2 0.1
       3 0 0.5
       3 1 3
 
 === AttriFile format ===
  FirstLine: AttributesCount<integer>
  Remaining: NodeID<integer> AttriIndex<integer>:AttriValue<float/integer> ...
  Note: unspecified entries will be set to 0.0
  
  e.g. 1606
       41407 34:1 33:1 32:1 31:1 27:1 28:8 29:1 30:1
       41380 17:240 16:1 114:2 8:2250 7:1 14:1 60:1 0:1 121:1 120:3 15:35 61:2 9:1 12:12 13:1
  e.g. 234
       2 5:0.85 6:-1.43 7:1.84 8:5.64 10:9.27 11:9.18
       1 0:1.79 1:1.79 2:0.00 3:0.00 4:1.00 5:1.00 6:-2.83
 
 === Output format ===
  FileName: attrirank_(DampingFactor).txt / attrirank_total.txt
  EachLine: NodeID AttriRankScore
 
 === Miscellaneous ===
  This implementation uses L1-Norm to check convergence, NodeCount * ConvergenceThreshold.
 
========================================================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <valarray>
#include <forward_list>
#include <unordered_map>
#define MAXLINELEN 8000
#define MAXPATHLEN 160

struct Node {
	static std::forward_list<Node *> dangle;
	static std::forward_list<Node *> normal;
	static std::valarray<double> piNew;
	static std::valarray<double> piOld;
	int id;
	std::unordered_map<Node *, double> outedge;
	std::valarray<double> attriVector;
	double *xOld;
	double *xNew;
	
	Node(int &i, int &count, std::forward_list<Node *> &nodeList): id(i) {
		attriVector = std::valarray<double>(0.0, count);
		nodeList.push_front(this);
	}
	~Node() {}
	void setTransition(void) {
		xOld = &piOld[id];
		xNew = &piNew[id];
		if (outedge.empty()) {
			dangle.push_front(this);
		} else {
			normal.push_front(this);
			double sum = 0.0;
			for (auto &x: outedge)
				sum += x.second;
			for (auto &x: outedge)
				x.second /= sum;
		}
		return;
	}
};
std::forward_list<Node *> Node::dangle;
std::forward_list<Node *> Node::normal;
std::valarray<double> Node::piNew;
std::valarray<double> Node::piOld;

void runAttriRank(const std::valarray<double> &provec, const double &damp, const int &maxiter, const double threshold) {
	printf("\tDampingFactor: %.2f\n", damp);
	if (damp == 0.0) {
		Node::piNew = provec;
		return;
	}
	Node::piOld = 1.0 / static_cast<double>(provec.size());
	for (int iteration = 1; iteration <= maxiter; ++iteration) {
		const double&& dangleSum = [] () { double s = 0.0; for (Node *v: Node::dangle) s += *v->xOld; return s; } ();
		Node::piNew = (dangleSum * damp + (1.0 - damp)) * provec;
		for (Node *v: Node::normal) {
			const double&& dampScore = damp * *v->xOld;
			for (auto &x: v->outedge)
				*x.first->xNew += x.second * dampScore;
		}
		const double&& err = std::abs(Node::piNew - Node::piOld).sum();
		if (err < threshold)	return;
		Node::piOld = Node::piNew;
	}
	printf("\t\tfailed to converge in %d iterations.\n", maxiter);
	return;
}

void runTotalRank(const std::valarray<double> &provec, const int &alpha, const int &beta, const int &maxiter, const double threshold) {
	printf("\tTotalRank: (alpha=%d, beta=%d)\n", alpha, beta);
	// Node::piOld is used as pho_current in this approach.
	Node::piNew = Node::piOld = (static_cast<double>(beta) / static_cast<double>(alpha + beta)) * provec;
	for (int iteration = 1; iteration <= maxiter; ++iteration) {
		const double&& dangleSum = [] () { double s = 0.0; for (Node *v: Node::dangle) s += *v->xOld; return s; } ();
		std::valarray<double>&& pho = dangleSum * provec;
		for (Node *v: Node::normal) {
			for (auto &x: v->outedge)
				pho[x.first->id] += x.second * *v->xOld;
		}
		pho *= static_cast<double>(iteration + alpha - 1) / static_cast<double>(iteration + alpha + beta);
		Node::piNew += pho;
		const double&& err = pho.sum();
		if (err < threshold)	return;
		Node::piOld  = pho;
	}
	printf("\t\tfailed to converge in %d iterations.\n", maxiter);
	return;
}

void outputFile(const char *fileName, std::forward_list<Node *> &nodeList) {
	nodeList.sort([] (Node *a, Node *b) { return (*a->xNew > *b->xNew); });
	FILE *fp = fopen(fileName, "w");
	for (Node *v: nodeList)
		fprintf(fp, "%d %e\n", v->id, *v->xNew);
	fclose(fp);
	return;
}

inline int wrongFormat(char *opt) {
	printf(">>> option '%s' needs parameter(s)\n", opt);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		printf(">>> The program needs at least 2 arguments: EdgeFile & AttriFile\n");
		return 0;
	}
	char argKernel[MAXPATHLEN] = "rbf_ap";
	bool unweighted = false;
	bool undirected = false;
	double converg = 1.0e-6;
	double damp[3] = { 0.0, 0.2, 1.0 };
	int param[2] = { 1, 1 };
	int maxiter = 100;
	for (int i = 3; i < argc; ++i) {
		if ((strcmp("-d", argv[i]) == 0) or (strcmp("--damp", argv[i]) == 0)) {
			if (++i >= argc)	return wrongFormat(argv[i - 1]);
			if (3 != sscanf(argv[i], "%lf,%lf,%lf", &damp[0], &damp[1], &damp[2])) {
				damp[0] = 0.0;	damp[1] = 0.2;	damp[2] = 1.0;
			}
			if (damp[0] < 0.0)	damp[0] = 0.0;
			if (damp[2] > 1.0)	damp[2] = 1.0;
		} else if (strcmp("--unweighted", argv[i]) == 0) {
			unweighted = true;
		} else if (strcmp("--undirected", argv[i]) == 0) {
			undirected = true;
		} else if ((strcmp("-k", argv[i]) == 0) or (strcmp("--kernel", argv[i]) == 0)) {
			if (++i >= argc)	return wrongFormat(argv[i - 1]);
			strncpy(argKernel, argv[i], MAXPATHLEN - 1);
		} else if ((strcmp("-i", argv[i]) == 0) or (strcmp("--iter", argv[i]) == 0)) {
			if (++i >= argc)	return wrongFormat(argv[i - 1]);
			maxiter = atoi(argv[i]);
			if (maxiter < 0)	maxiter = 100;
		} else if ((strcmp("-c", argv[i]) == 0) or (strcmp("--conv", argv[i]) == 0)) {
			if (++i >= argc)	return wrongFormat(argv[i - 1]);
			converg = atof(argv[i]);
			if (converg < 0)	converg = 1.0e-6;
		} else if ((strcmp("-t", argv[i]) == 0) or (strcmp("--total", argv[i]) == 0)) {
			if (++i >= argc)	return wrongFormat(argv[i - 1]);
			if (2 != sscanf(argv[i], "%d,%d", &param[0], &param[1])) {
				param[0] = 1;	param[1] = 1;
			}
			if (param[0] < 0)	param[0] = 1;
			if (param[1] < 0)	param[1] = 1;
		} else {
			printf("\tunknown argument: %s\n", argv[i]);
		}
	}
	printf("[GraphType] %s + %s\n", unweighted ? "unweighted" : "weighted", undirected ? "undirected" : "directed");
	printf("[MaxIterations] %d\n", maxiter);
	printf("[ConvThreshold] %.2e\n", converg);
	// args parse end
	std::unordered_map<int, Node *> nodes;
	std::forward_list<Node *> nodeList;
	/* AttriFile */
	int attriCount;
	{
		int u, a, s, i;
		double f;	char buff[MAXLINELEN];
		FILE *fp = fopen(argv[2], "r");
		fgets(buff, MAXLINELEN - 1, fp);
		sscanf(buff, "%d", &attriCount);
		printf("AttriCount: %d\n", attriCount);
		while (fgets(buff, MAXLINELEN - 1, fp) != NULL) {
			sscanf(buff, "%d%n", &u, &i);
			if (nodes.count(u) == 0)	nodes[u] = new Node(u, attriCount, nodeList);
			while (sscanf(buff + i, "%d:%lf%n", &a, &f, &s) == 2) {
				nodes[u]->attriVector[a] = f;
				i += s;
			}
		}
		fclose(fp);
	}
	/* EdgeFile */
	{
		int u, v, arg;
		double w;	char buff[MAXLINELEN];
		FILE *fp = fopen(argv[1], "r");
		while (fgets(buff, MAXLINELEN - 1, fp) != NULL) {
			arg = sscanf(buff, "%d %d %lf", &u, &v, &w);
			if (nodes.count(u) == 0)	nodes[u] = new Node(u, attriCount, nodeList);
			if (nodes.count(v) == 0)	nodes[v] = new Node(v, attriCount, nodeList);
			if (unweighted) {
				nodes[u]->outedge[nodes[v]] = 1.0;
				if (undirected)
				nodes[v]->outedge[nodes[u]] = 1.0;
			} else {
				if (arg == 2)	w = 1.0;
				nodes[u]->outedge[nodes[v]] += w;
				if (undirected)
				nodes[v]->outedge[nodes[u]] += w;
			}
		}
		fclose(fp);
	}
	const double&& nodeCount = static_cast<double>(nodes.size());
	/* Standardization */
	for (int i = 0; i < attriCount; ++i) {
		double e1x = 0.0;
		double e2x = 0.0;
		for (Node *v: nodeList) {
			e1x += v->attriVector[i];
			e2x += v->attriVector[i] * v->attriVector[i];
		}
		const double&& mean = e1x / nodeCount;
		const double&& std = std::sqrt(e2x / nodeCount - mean * mean);
		for (Node *v: nodeList)
			v->attriVector[i] = (std > 0.0) ? ((v->attriVector[i] - mean) / std) : 0.0;
	}
	/* TransitionMatrix */
	printf("Generate Transition Matrix\n");
	Node::piNew = std::valarray<double>(nodes.size());
	Node::piOld = std::valarray<double>(nodes.size());
	for (Node *v: nodeList)
		v->setTransition();
	const double&& gamma = 1.0 / attriCount;
	std::valarray<double> resetVec(0.0, nodes.size());
	int transProcess = static_cast<int>(nodes.size());
	if (strcmp("rbf", argKernel) == 0) {
		printf("\tusing RBF kernel\n");
		for (auto it1 = nodeList.begin(); it1 != nodeList.end(); ++it1) {
			resetVec[(*it1)->id] += 1.0;
			for (auto it2 = std::next(it1); it2 != nodeList.end(); ++it2) {
				double&& s12 = [&gamma] (const Node *a, const Node *b) {
					return std::exp(-gamma * std::pow(a->attriVector - b->attriVector, 2.0).sum());
				} (*it1, *it2);
				resetVec[(*it1)->id] += s12;
				resetVec[(*it2)->id] += s12;
			}
			printf("\r\tremain: %7d", --transProcess);
		}
	} else if (strcmp("cosine", argKernel) == 0) {
		printf("\tusing Cosine similarity kernel\n");
		std::valarray<double> unitSum(0.0, attriCount);
		for (Node *v: nodeList) {
			v->attriVector /= std::sqrt(std::pow(v->attriVector, 2.0).sum());
			unitSum += v->attriVector;
			printf("\r\tremain(1/2): %7d", --transProcess);
		}
		transProcess = static_cast<int>(nodes.size());
		for (Node *v: nodeList) {
			resetVec[v->id] = ((v->attriVector * unitSum).sum() + nodeCount) / 2.0;
			printf("\r\tremain(2/2): %7d", --transProcess);
		}
	} else {
		printf("\tusing RBF kernel (approximation)\n");
		std::valarray<double> scalarW(nodes.size());
		for (Node *v: nodeList)
			scalarW[v->id] = std::exp(-gamma * std::pow(v->attriVector, 2.0).sum());
		std::valarray<double> vectorB(0.0, attriCount);
		std::valarray<double> *matrixC = new std::valarray<double>[attriCount];
		for (int i = 0; i < attriCount; ++i)
			matrixC[i] = std::valarray<double>(0.0, attriCount);
		for (Node *v: nodeList) {
			std::valarray<double>&& wx = scalarW[v->id] * v->attriVector;
			vectorB += wx;
			for (int i = 0; i < attriCount; ++i)
				matrixC[i] += wx * v->attriVector[i];
			printf("\r\tremain(1/2): %7d", --transProcess);
		}
		vectorB *= 2.0 * gamma;
		for (int i = 0; i < attriCount; ++i)
			matrixC[i] *= 2.0 * gamma * gamma;
		double&& scalarA = scalarW.sum();
		transProcess = static_cast<int>(nodes.size());
		for (Node *v: nodeList) {
			std::valarray<double> Cx(attriCount);
			for (int i = 0; i < attriCount; ++i)
				Cx[i] = (matrixC[i] * v->attriVector).sum();
			resetVec[v->id] = scalarW[v->id] * (scalarA + (v->attriVector * (vectorB + Cx)).sum());
			printf("\r\tremain(2/2): %7d", --transProcess);
		}
		delete[] matrixC;
	}
	putchar('\n');	resetVec /= resetVec.sum();
	/* AttriRank */
	printf("Run AttriRank Model\n");
	for (double df = damp[0]; df <= damp[2]; df += damp[1]) {
		runAttriRank(resetVec, df, maxiter, nodeCount * converg);
		char fileName[40];
		sprintf(fileName, "attrirank_%.3f.txt", df);
		outputFile(fileName, nodeList);
	}
	runTotalRank(resetVec, param[0], param[1], maxiter, nodeCount * converg);
	outputFile("attrirank_total.txt", nodeList);
	return 0;
}
