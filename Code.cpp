#include <torch/torch.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <chrono>
#include <map>
#include <random>
#include <iostream>
#include <algorithm> 
#include <fstream>
#include <thread>
#include <float.h>
#include <vector>
#include <time.h>
#include <math.h>
#define M (1000)
#define MAXL1CHILD (10000)
#define LIMIT_M_2 (M % 2 ? (M + 1) / 2 : M / 2)

using namespace std;
using namespace chrono;
ofstream ofs;
int *tolbits = new int;
float bEST=0.01;
typedef struct BPlusNode *BPlusTree, *Position;
typedef bool *ZtupleBin;
typedef struct MADENet
{
	int zdr;
	int zdc;
	int connectlen;
	int leafnums = 100;
	int diglen;
	float *fc1w;
	float *fc1b;
	float *fc2w;
	float *fc2b;
	void *Torchnet = NULL;
} MADENet;

typedef struct ZTuple
{
	ZtupleBin bin;
	long long *values;
	long long z32;
} ZTuple;

typedef struct ZTab
{
	int r;
	int c;
	vector<ZTuple *> D;
} ZTab;
typedef ZTuple *KeyType;

bool ZTcmp(ZTuple *v1, ZTuple *v2)
{
	return v1->z32 < v2->z32;
}

typedef struct LinearFunc
{
	double x0; // normalized
	double y0;
	double kup;
	double kdown;
} linf;

struct BPlusNode
{
	int KeyNum;
	int linNum = 0;
	long long preNum = 0;
	KeyType *Key;
	linf *LinearX;
	long long Zmax;
	long long Zmin;
	BPlusTree *Children;
	BPlusTree Next;
};

typedef struct Query
{
	int queryid;	  // Ψһ����Queryid
	int columnNumber; // how many columns
	int *binaryLength;
	long long *leftupBound;
	long long *rightdownBound;
} Query;
typedef struct Querys
{
	Query *Qs;
	int queryNumber;
} Querys;
map<int, long> qid2TrueNumber;

MADENet *MadeBuffer[20];
Query *QueryBuffer[20];
float tret[20];

BPlusNode *lastN = NULL;
typedef struct MADE2BPlus
{
	BPlusTree transferLayer[MAXL1CHILD];
	bool Flag[MAXL1CHILD];
	int curnum = 200;
} MiddleLayer;

typedef struct CardIndex
{
	MADENet *Mnet;
	MiddleLayer *trans;
	BPlusNode *Head;
} CardIndex;

MADENet *loadMade(string filePath)
{
	ifstream infile(filePath);
	if (!infile.is_open())
	{
		cout << "Fail to Net load Tree" << endl;
		return NULL;
	}
	int bittol;
	// infile >> bittol;
	// cout<<"bt:"<<bittol<<endl;
	MADENet *ret = new MADENet;
	infile >> ret->zdr >> ret->zdc >> ret->connectlen >> ret->leafnums;
	bittol = ret->zdc;
	ret->diglen = bittol;
	ret->fc1w = new float[bittol * bittol];
	ret->fc2w = new float[bittol * bittol];
	ret->fc1b = new float[bittol];
	ret->fc2b = new float[bittol];
	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			infile >> ret->fc1w[i * bittol + j];
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		infile >> ret->fc1b[i];
	}
	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			infile >> ret->fc2w[i * bittol + j];
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		infile >> ret->fc2b[i];
	}
	int strcord = 0;
	infile.close();
	return ret;
}

/* ��ʼ�� */
extern BPlusTree Initialize();
/* ���� */
extern BPlusTree Insert(BPlusTree T, KeyType Key);
/* ɾ�� */
// extern BPlusTree Remove(BPlusTree T,KeyType Key);
/* ���� */
extern BPlusTree Destroy(BPlusTree T);
/* �����ڵ� */
extern void Travel(BPlusTree T);
/* ������Ҷ�ڵ������ */
extern void TravelData(BPlusTree T);
float cdfCalculate(MADENet *Mnet, ZTuple *ztup);
void maintainLinearNode(BPlusNode *ptr);
void maintainBPlusProperty(BPlusNode *Head);
int LinearNodeSearch(BPlusNode *subN, ZTuple *ZT0);
int linearNodeSearchPre(BPlusNode *subN, ZTuple *ZT0);
int rangeQueryExceuteF(BPlusTree T, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1);
BPlusNode *linearLeafSearch(BPlusNode *subN, ZTuple *ZT0);
BPlusNode *CardIndexLeafGet(CardIndex *C, ZTuple *ZT0);
KeyType Unavailable = NULL;

default_random_engine e;
uniform_real_distribution<float> u(0.0, 1.0);
float midlle[30];
int infL = 30;
long long mallocedSize = 0;
long long overallSize = 0;
int colPattern[100];
vector<vector<int>> Zidx2Col;
int randG(float oneProb)
{
	if (u(e) <= oneProb)
	{
		return 1;
	}
	return 0;
}

void longlong2digVec(long long valx, int *vx, int diglen)
{
	for (int i = 0; i < diglen; i++)
	{
		vx[diglen - i - 1] = (valx % 2);
		valx /= 2;
	}
	if (valx != 0)
	{ // overflow
		// cout << "overflow" << endl;
		for (int i = 0; i < diglen; i++)
		{
			vx[diglen - i - 1] = 1;
		}
	}
}
class MaskedLinearImpl : public torch::nn::Module
{
public:
	torch::Tensor mask;
	torch::Tensor weight;
	torch::Tensor bias;
	MaskedLinearImpl(int64_t in_features, int64_t out_features, bool bias = true)
	{
		this->mask = register_buffer("mask", torch::ones({out_features, in_features}));
		this->weight = register_parameter("weight", torch::ones({out_features, in_features}));
		this->bias = register_parameter("bias", torch::ones({out_features}));
	}

	void set_mask(torch::Tensor new_mask)
	{
		mask.data().copy_(new_mask.t());
	}

	torch::Tensor forward(torch::Tensor input)
	{
		return torch::nn::functional::linear(input, mask * weight, bias);
	}
};

TORCH_MODULE(MaskedLinear);

class MyMADEImpl : public torch::nn::Module
{
public:
	MaskedLinear fc1 = nullptr;
	MaskedLinear fc3 = nullptr;
	int64_t Xl;
	int64_t iD;
	int64_t lS;
	MyMADEImpl(int64_t Xlength = 6, int64_t innnerDepth = 3, int64_t linearScaleN = 5)
	{
		innnerDepth = Xlength;
		// mk : order
		std::vector<std::vector<int64_t>> MK;
		std::vector<int64_t> m0;
		std::vector<int64_t> m3;
		Xl = Xlength;
		iD = innnerDepth;
		lS = linearScaleN;
		for (int64_t i = 0; i < Xlength; ++i)
		{
			m0.push_back(i);
			m3.push_back(i);
		}
		std::vector<int64_t> m1;
		for (int64_t i = 0; i < innnerDepth; ++i)
		{
			m1.push_back(i);
		}
		MK.push_back(m0);
		MK.push_back(m1);
		MK.push_back(m3);
		std::vector<std::vector<int64_t>> iolengthList = {{Xlength, innnerDepth}, {innnerDepth, Xlength}};
		int64_t idx = 0;
		for (const auto &L : iolengthList)
		{
			idx += 1;
			int64_t i0 = L[0];
			int64_t j0 = L[1];
			torch::Tensor mask = torch::zeros({i0, j0});
			for (int64_t i = 0; i < i0; ++i)
			{
				for (int64_t j = 0; j < j0; ++j)
				{
					const std::vector<int64_t> &maskp0 = MK[idx - 1];
					const std::vector<int64_t> &maskp1 = MK[idx];
					if (maskp0[i] < maskp1[j] && maskp0[i] >= (maskp1[j] - linearScaleN))
					{
						mask[i][j] = 1;
					}
					else
					{
						mask[i][j] = 0;
					}
				}
			}
			maskList.push_back(mask);
		}
		fc1 = register_module("fc1", MaskedLinear(Xlength, innnerDepth));
		fc1->set_mask(maskList[0]);
		fc3 = register_module("fc3", MaskedLinear(innnerDepth, Xlength));
		fc3->set_mask(maskList[1]);
	}

public:
	torch::Tensor forward(torch::Tensor x)
	{
		x = x.view({x.size(0), -1}).to(torch::kFloat32);
		x = torch::relu(fc1->forward(x));
		x = torch::sigmoid(fc3->forward(x));
		return x;
	}

	std::vector<torch::Tensor> maskList;
};

TORCH_MODULE(MyMADE);
MyMADE *Core = NULL;

void UpdateMymade(MyMADE libnet, int zdr, MADENet *ret)
{
	MyMADEImpl MI = *(*Core).get();
	int bittol = ret->zdc;
	ret->zdr += zdr;
	MaskedLinearImpl MLI = *MI.fc1.get();
	MaskedLinearImpl MLII = *MI.fc3.get();
	torch::Tensor maskI = MLI.mask.cpu();
	torch::Tensor weightI = MLI.weight.cpu();
	torch::Tensor biasI = MLI.bias.cpu();
	torch::Tensor maskII = MLII.mask.cpu();
	torch::Tensor weightII = MLII.weight.cpu();
	torch::Tensor biasII = MLII.bias.cpu();

	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			ret->fc1w[i * bittol + j] = maskI[i][j].item<float>() * weightI[i][j].item<float>();
			ret->fc2w[i * bittol + j] = maskII[i][j].item<float>() * weightII[i][j].item<float>();
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		ret->fc1b[i] = biasI[i].item<float>();
		ret->fc2b[i] = biasII[i].item<float>();
	}
}

MADENet *libT2Mymade(MyMADE libnet, int zdr)
{
	MADENet *ret = new MADENet;
	int mallocSize = 0;
	MyMADEImpl MI = *libnet.get();
	ret->zdr = zdr;
	ret->zdc = MI.Xl;
	ret->connectlen = MI.lS;
	int bittol = ret->zdc;
	ret->diglen = bittol;
	ret->fc1w = new float[bittol * bittol];
	ret->fc2w = new float[bittol * bittol];
	ret->fc1b = new float[bittol];
	ret->fc2b = new float[bittol];
	mallocSize += sizeof(float) * (bittol * bittol * 2 + bittol * 2);
	cout << "Core NetSize(KB):" << mallocSize / 1000 << endl;
	ofs << "Core NetSize(KB):" << mallocSize / 1000 << endl;
	MaskedLinearImpl MLI = *MI.fc1.get();
	MaskedLinearImpl MLII = *MI.fc3.get();
	torch::Tensor maskI = MLI.mask.cpu();
	torch::Tensor weightI = MLI.weight.cpu();
	torch::Tensor biasI = MLI.bias.cpu();

	torch::Tensor maskII = MLII.mask.cpu();
	torch::Tensor weightII = MLII.weight.cpu();
	torch::Tensor biasII = MLII.bias.cpu();

	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			ret->fc1w[i * bittol + j] = maskI[i][j].item<float>() * weightI[i][j].item<float>();
			ret->fc2w[i * bittol + j] = maskII[i][j].item<float>() * weightII[i][j].item<float>();
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		ret->fc1b[i] = biasI[i].item<float>();
		ret->fc2b[i] = biasII[i].item<float>();
	}
	ret->Torchnet = (void *)&libnet;
	return ret;
}
MADENet *IncrementalTraining(torch::Tensor *Dptr, torch::Tensor *Iptr, MADENet *Net, int DO = 0, int DI = 0)
{
	torch::Tensor D = *Dptr;
	torch::Tensor I = *Iptr;
	int r1 = D.size(0);
	int r2 = I.size(0);
	int minr = std::min(r2, r1);
	cout << r1 << " " << r2 << endl;
	torch::Tensor idxPerm = torch::randperm(minr);
	std::vector<std::vector<int64_t>> idxList;
	int64_t batchS = 1024;
	for (int ix = 0; ix < minr / batchS; ix++)
	{
		idxList.push_back({ix * batchS, min((int)minr, (int)((ix + 1) * batchS))});
	}
	int64_t lastv = idxList.back()[1];
	if (lastv < minr)
	{
		idxList.push_back({lastv, minr});
	}
	torch::nn::BCELoss losfunc;
	MyMADE net;
	if (Core == NULL)
	{
		int64_t fetureNum = D.sizes()[1];
		int64_t hidNum = fetureNum;
		int linearScaleN = 30;
		Core = new MyMADE(fetureNum, hidNum, linearScaleN);
		net = *Core;
	}
	else
	{
		net = *(Core);
	}
	torch::Device device(torch::kCUDA); // Use torch::kCPU if CUDA is not available
	net->to(device);
	torch::optim::Adam optimizer(net->parameters());
	int batch_idx = 0;
	float tdata1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	float ty = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	int trainingIter = 1;
	for (int iii = 0; iii < trainingIter; iii++)
	{
		for (const auto &range : idxList)
		{
			if (DI == 0 && DO == 0)
			{
				int64_t startidx0 = range[0];
				int64_t endidx0 = range[1];
				torch::Tensor x = I.index_select(0, idxPerm.slice(0, startidx0, endidx0)).to(torch::kCUDA);
				optimizer.zero_grad();
				torch::Tensor out = net->forward(x + 0.0);
				torch::Tensor l1 = losfunc(out, x + 0.0) * (r2) / (r1 + r2);
				torch::Tensor ipt = D.index_select(0, idxPerm.slice(0, startidx0, endidx0)).to(torch::kCUDA);
				out = net->forward(ipt + 0.0);
				torch::Tensor l2 = losfunc(out, ipt + 0.0) * (r1) / (r1 + r2);
				torch::Tensor loss = l1 + l2;
				loss.backward();
				optimizer.step();
				batch_idx += 1;
				if (batch_idx % 100 == 0)
				{
					std::cout << "\r" << batch_idx << "/" << (idxList.size() * trainingIter) << " loss: " << loss.item<float>() << " ";
					std::cout.flush();
				}
			}
			else
			{
				int64_t startidx0 = range[0];
				int64_t endidx0 = range[1];
				torch::Tensor x = I.index_select(0, idxPerm.slice(0, startidx0, endidx0)).to(torch::kCUDA);
				optimizer.zero_grad();
				torch::Tensor out = net->forward(x + 0.0);
				torch::Tensor l1 = losfunc(out, x + 0.0) * (DI) / (DI + DO);
				torch::Tensor ipt = D.index_select(0, idxPerm.slice(0, startidx0, endidx0)).to(torch::kCUDA);
				out = net->forward(ipt + 0.0);
				torch::Tensor l2 = losfunc(out, ipt + 0.0) * (DO) / (DI + DO);
				torch::Tensor loss = l1 + l2;
				loss.backward();
				optimizer.step();
				batch_idx += 1;
				if (batch_idx % 100 == 0)
				{
					std::cout << "\r" << batch_idx << "/" << (idxList.size() * trainingIter) << " loss: " << loss.item<float>() << " ";
					std::cout.flush();
				}
			}
		}
	}
	float tx = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	if (Net != NULL)
	{
		UpdateMymade(net, r2, Net);
	}
	else
	{
		MADENet *ret = libT2Mymade(net, D.sizes()[0] + I.sizes()[0]);
		return ret;
	}
	return Net;
}

int reload = 0;
MADENet *smallDataTraining(torch::Tensor *Dptr, torch::Tensor *Iptr, int64_t linearScaleN, int loopIter)
{
	// ����batchS = 1024 ����I��Train
	// Ȼ���ô�batch��D��Trainһ��Epoch
	torch::Tensor D = *Dptr;
	torch::Tensor I = *Iptr;
	cout << "Size info";
	cout << D.sizes()[0] << " " << I.sizes()[0] << endl;
	int64_t fetureNum = D.sizes()[1];
	int64_t hidNum = fetureNum;
	int64_t trainingIter = 20;
	int r1 = I.size(0);
	int minr = r1;
	torch::Tensor idxPerm = torch::randperm(minr);
	std::vector<std::vector<int64_t>> idxList;
	int64_t batchS = 1024;
	for (int ix = 0; ix < minr / batchS; ix++)
	{
		idxList.push_back({ix * batchS, min((int)minr, (int)((ix + 1) * batchS))});
	}
	int64_t lastv = idxList.back()[1];
	if (lastv < minr)
	{
		idxList.push_back({lastv, minr});
	}
	cout << fetureNum << " " << hidNum << " " << linearScaleN << endl;
	Core = new MyMADE(fetureNum, hidNum, linearScaleN);
	MyMADE net;
	reload = 0;
	if (reload)
	{
		torch::load(*(Core), "./model.pt"); //*(Core);
		net = *Core;
	}
	else
	{
		net = *Core;
	}
	torch::Device device(torch::kCUDA); // Use torch::kCPU if CUDA is not available
	net->to(device);
	cout << device.str() << endl;
	std::cout << "cold start On small batch" << std::endl;
	torch::nn::BCELoss losfunc;
	torch::optim::Adam optimizer(net->parameters());
	int64_t batch_idx = 0;
	std::cout << "Data preparationD " << std::endl;
	auto ty = std::chrono::steady_clock::now();
	for (int64_t iii = 0; iii < trainingIter; iii++)
	{
		for (const auto &idx : idxList)
		{
			if (idx.size() <= 1)
			{
				continue;
			}
			optimizer.zero_grad();
			torch::Tensor ipt = I.index_select(0, idxPerm.slice(0, idx[0], idx[1])).to(torch::kCUDA);
			torch::Tensor out = net->forward(ipt + 0.0);
			torch::Tensor l2 = losfunc(out, ipt + 0.0);
			torch::Tensor loss = l2;
			loss.backward();
			optimizer.step();
			batch_idx += 1;
			if (batch_idx % 100 == 0)
			{
				std::cout << "\r" << batch_idx << "/" << (idxList.size() * trainingIter) << " loss: " << loss.item<float>() << " ";
				std::cout.flush();
			}
		}
	}
	auto tx = std::chrono::steady_clock::now();
	std::cout << "\nSmall batch Training Takes: " << std::chrono::duration_cast<std::chrono::seconds>(tx - ty).count() << std::endl;

	int r2 = D.size(0);
	minr = r2;
	idxPerm = torch::randperm(minr);
	std::vector<std::vector<int64_t>> idxList2;
	batchS = 4096;
	for (int ix = 0; ix < minr / batchS; ix++)
	{
		idxList2.push_back({ix * batchS, min((int)minr, (int)((ix + 1) * batchS))});
	}
	lastv = idxList2.back()[1];
	if (lastv < minr)
	{
		idxList2.push_back({lastv, minr});
	}
	net->to(device);
	std::cout << "hot start On BIG batch" << std::endl;
	batch_idx = 0;
	std::cout << "Data preparationD " << std::endl;
	ty = std::chrono::steady_clock::now();
	trainingIter = 1;
	for (int64_t iii = 0; iii < trainingIter; iii++)
	{
		for (const auto &idx : idxList2)
		{
			if (idx.size() <= 1)
			{
				continue;
			}
			optimizer.zero_grad();
			torch::Tensor ipt = D.index_select(0, idxPerm.slice(0, idx[0], idx[1])).to(torch::kCUDA);
			torch::Tensor out = net->forward(ipt + 0.0);
			torch::Tensor l2 = losfunc(out, ipt + 0.0);
			torch::Tensor loss = l2;
			loss.backward();
			optimizer.step();
			batch_idx += 1;
			if (batch_idx % 100 == 0)
			{
				std::cout << "\r" << batch_idx << "/" << (idxList.size() * trainingIter) << " loss: " << loss.item<float>() << " ";
				std::cout.flush();
			}
		}
	}
	tx = std::chrono::steady_clock::now();
	std::cout << "\BIG batch Training Takes: " << std::chrono::duration_cast<std::chrono::seconds>(tx - ty).count() << std::endl;
	MADENet *ret = libT2Mymade(net, D.sizes()[0]);
	torch::save(net, "./model.pt");
	return ret;
}
MADENet *TrainOnD(torch::Tensor *Dptr, int64_t linearScaleN, int loopIter)
{
	torch::Tensor D = *Dptr;
	int64_t fetureNum = D.sizes()[1];
	int64_t hidNum = fetureNum;
	int64_t trainingIter = loopIter;
	int r1 = D.size(0);
	int minr = r1;
	torch::Tensor idxPerm = torch::randperm(minr);
	std::vector<std::vector<int64_t>> idxList;
	int64_t batchS = 2048;
	for (int ix = 0; ix < minr / batchS; ix++)
	{
		idxList.push_back({ix * batchS, min((int)minr, (int)((ix + 1) * batchS))});
	}
	int64_t lastv = idxList.back()[1];
	if (lastv < minr)
	{
		idxList.push_back({lastv, minr});
	}
	cout << fetureNum << " " << hidNum << " " << linearScaleN << endl;
	Core = new MyMADE(fetureNum, hidNum, linearScaleN);
	MyMADE net;
	reload = 0;
	if (reload)
	{
		torch::load(*(Core), "./model.pt"); //*(Core);
		net = *Core;
	}
	else
	{
		net = *Core;
	}
	torch::Device device(torch::kCUDA); // Use torch::kCPU if CUDA is not available
	net->to(device);
	cout << device.str() << endl;
	std::cout << "cold start...." << std::endl;
	torch::nn::BCELoss losfunc;
	torch::optim::Adam optimizer(net->parameters());
	int64_t batch_idx = 0;
	std::cout << "Data preparationD " << std::endl;
	auto ty = std::chrono::steady_clock::now();
	cout << "Moving D to GPU" << endl;
	long long moveT = 0;
	long long calTime = 0;
	for (int64_t iii = 0; iii < trainingIter; iii++)
	{
		for (const auto &idx : idxList)
		{
			if (idx.size() <= 1)
			{
				continue;
			}
			optimizer.zero_grad();
			// auto tpreMove = std::chrono::steady_clock::now();
			torch::Tensor ipt = D.index_select(0, idxPerm.slice(0, idx[0], idx[1])).to(torch::kCUDA);
			// auto tpostMove = std::chrono::steady_clock::now();
			torch::Tensor out = net->forward(ipt + 0.0);
			torch::Tensor l2 = losfunc(out, ipt + 0.0);
			torch::Tensor loss = l2;
			loss.backward();
			optimizer.step();
			// auto tpostCalculation = std::chrono::steady_clock::now();
			// moveT += std::chrono::duration_cast<std::chrono::nanoseconds>(tpostMove - tpreMove).count();
			// calTime+= std::chrono::duration_cast<std::chrono::nanoseconds>(tpostCalculation - tpostMove).count();
			batch_idx += 1;
			if (batch_idx % 100 == 0)
			{
				std::cout << "\r" << batch_idx << "/" << (idxList.size() * trainingIter) << " loss: " << loss.item<float>() << " ";
				std::cout.flush();
			}
		}
	}
	auto tx = std::chrono::steady_clock::now();
	cout << "MoveT:" << moveT << " CalT" << calTime << endl;
	std::cout << "\nTraining TKS: " << std::chrono::duration_cast<std::chrono::seconds>(tx - ty).count() << std::endl;
	MADENet *ret = libT2Mymade(net, D.sizes()[0]);
	torch::save(net, "./model.pt");
	return ret;
}

ZTuple *makeZT(bool *zx, int binaryLen)
{
	ZTuple *ZT0 = new ZTuple;
	ZT0->bin = zx;
	long long z32 = 0;
	for (int j = 0; j < min(binaryLen, 60); j++)
	{
		z32 *= 2;
		z32 += zx[j];
	}
	ZT0->z32 = z32;
	return ZT0;
}

ZTab *loadZD(string filePath)
{
	cout << "Start2Load " << endl;
	ZTab *ZT = new ZTab;
	ifstream infile(filePath);
	int r, c;
	int r2, c2;
	infile >> r >> c >> r2 >> c2;
	cout << "Rows:" << r << " Cols:" << c << endl;
	ZT->r = r;
	ZT->c = c;
	for (int i = 0; i < r; i++)
	{
		// cout<<"\r LOADING"<<i;
		ZtupleBin zx = new bool[c];
		// ZTuple *ZT0 = new ZTuple;
		for (int j = 0; j < c; j++)
		{
			bool v;
			infile >> v;
			zx[j] = v;
		}
		ZTuple *ZT0 = makeZT(zx, c);
		ZT0->values = new long long[c];
		for (int j = 0; j < c2; j++)
		{
			infile >> ZT0->values[j];
		}
		ZT->D.push_back(ZT0);
	}
	// cout<<endl;
	infile.close();
	sort(ZT->D.begin(), ZT->D.end(), ZTcmp);
	return ZT;
}
ZTab *loadZDBin(string filePath, int OSM = 0, int row = 0)
{
	cout << "Start2Load " << endl;
	typedef std::chrono::high_resolution_clock Clock;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	ZTab *ZT = new ZTab;
	FILE *fp;
	if (fopen_s(&fp, filePath.c_str(), "rb") != 0)
	{
		cout << "Failed to open the file." << endl;
		return nullptr;
	}
	int r, c;
	int r2, c2;
	fread(&r, sizeof(int), 1, fp);
	fread(&c, sizeof(int), 1, fp);
	fread(&r2, sizeof(int), 1, fp);
	fread(&c2, sizeof(int), 1, fp);
	if (OSM)
	{
		r = row;
	}
	cout << "Rows:" << r << " Cols:" << c << endl;
	ZT->r = r;
	ZT->c = c;
	for (int i = 0; i < r; i++)
	{
		// cout<<"\r LOADING"<<i;
		ZtupleBin zx = new bool[c];
		// ZTuple *ZT0 = new ZTuple;
		for (int j = 0; j < c; j++)
		{
			bool v;
			fread(&v, sizeof(bool), 1, fp);
			// cout << v;
			zx[j] = v;
		} // cout << endl;
		ZTuple *ZT0 = makeZT(zx, c);
		ZT0->values = new long long[c];
		for (int j = 0; j < c2; j++)
		{
			fread(&ZT0->values[j], sizeof(long long), 1, fp);
			// cout << ZT0->values[j] << " ";
		} // cout << endl;
		ZT->D.push_back(ZT0);
	}
	// cout<<endl;
	fclose(fp);
	sort(ZT->D.begin(), ZT->D.end(), ZTcmp);
	auto t4 = Clock::now(); // ��ʱ��ʼ
	cout << "LoadData Time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) << endl;
	return ZT;
}

bool *loadZDBulk(string filePath, int OSM = 0)
{
	cout << "Start2Load " << endl;
	typedef std::chrono::high_resolution_clock Clock;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	ZTab *ZT = new ZTab;
	FILE *fp;
	if (fopen_s(&fp, filePath.c_str(), "rb") != 0)
	{
		cout << "Failed to open the file." << endl;
		return nullptr;
	}
	int r, c;
	int r2, c2;
	fread(&r, sizeof(int), 1, fp);
	fread(&c, sizeof(int), 1, fp);
	fread(&r2, sizeof(int), 1, fp);
	fread(&c2, sizeof(int), 1, fp);
	// if (OSM)
	// {
	// 	r = 40000000;
	// }
	cout << "Rows:" << r << " Cols:" << c << endl;
	ZT->r = r;
	ZT->c = c;
	cout << "Trying 2 malloc space:" << (r * c) / (1e9) << "GB" << endl;

	bool *p = (bool *)malloc((r * c));
	cout << p << endl;
	for (long long i = 0; i < r; i++)
	{
		for (long long j = 0; j < c; j++)
		{
			fread(&p[i * c + j], sizeof(bool), 1, fp);
		}
		for (int j = 0; j < c2; j++)
		{
			long long tmp;
			fread(&tmp, sizeof(long long), 1, fp);
			// cout << ZT0->values[j] << " ";
		} // cout << endl;
	}
	// cout<<endl;
	fclose(fp);
	auto t4 = Clock::now(); // ��ʱ��ʼ
	cout << "LoadData Time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) << endl;
	return p;
}
void transZD(string filePath, const char *outPath)
{
	cout << "Start2Load " << endl;
	ZTab *ZT = new ZTab;
	ifstream infile(filePath);
	FILE *fp;
	if (fopen_s(&fp, outPath, "wb") != 0)
	{
		cout << "Failed to open the file." << endl;
		return;
	}
	int r, c;
	int r2, c2;
	infile >> r >> c >> r2 >> c2;
	fwrite(&r, sizeof(int), 1, fp);
	fwrite(&c, sizeof(int), 1, fp);
	fwrite(&r2, sizeof(int), 1, fp);
	fwrite(&c2, sizeof(int), 1, fp);
	cout << "Rows:" << r << " Cols:" << c << endl;
	cout << c2 << " " << r2 << endl;
	ZT->r = r;
	ZT->c = c;
	for (int i = 0; i < r; i++)
	{
		// cout<<"\r LOADING"<<i;
		ZtupleBin zx = new bool[c];
		// ZTuple *ZT0 = new ZTuple;
		for (int j = 0; j < c; j++)
		{
			bool v;
			infile >> v;
			fwrite(&v, sizeof(bool), 1, fp);
			zx[j] = v;
		}
		ZTuple *ZT0 = makeZT(zx, c);
		ZT0->values = new long long[c];
		for (int j = 0; j < c2; j++)
		{
			infile >> ZT0->values[j];
			fwrite(&ZT0->values[j], sizeof(long long), 1, fp);
		}
		ZT->D.push_back(ZT0);
	}
	// cout<<endl;
	infile.close();
}
/* ���ɽڵ㲢��ʼ�� */
static BPlusTree MallocNewNode()
{
	BPlusTree NewNode;
	int i;
	NewNode = (BPlusTree)malloc(sizeof(struct BPlusNode));
	if (NewNode == NULL)
		exit(EXIT_FAILURE);
	NewNode->Key = new KeyType[M + 1];
	NewNode->LinearX = new linf[M + 1];
	NewNode->Children = new BPlusTree[M + 1];
	i = 0;
	mallocedSize += (M + 1) * (sizeof(KeyType) + sizeof(linf) + sizeof(BPlusTree));
	while (i < M + 1)
	{
		NewNode->Key[i] = Unavailable;
		NewNode->Children[i] = NULL;
		i++;
	}
	NewNode->Next = NULL;
	NewNode->KeyNum = 0;
	return NewNode;
}
/* ��ʼ�� */
extern BPlusTree Initialize()
{

	BPlusTree T;
	if (M < (3))
	{
		printf("M��С����3��");
		exit(EXIT_FAILURE);
	}
	/* ����� */
	T = MallocNewNode();

	return T;
}

static Position FindMostLeft(Position P)
{
	Position Tmp;

	Tmp = P;

	while (Tmp != NULL && Tmp->Children[0] != NULL)
	{
		Tmp = Tmp->Children[0];
	}
	return Tmp;
}

static Position FindMostRight(Position P)
{
	Position Tmp;

	Tmp = P;

	while (Tmp != NULL && Tmp->Children[Tmp->KeyNum - 1] != NULL)
	{
		Tmp = Tmp->Children[Tmp->KeyNum - 1];
	}
	return Tmp;
}

/* Ѱ��һ���ֵܽڵ㣬��洢�Ĺؼ���δ�������򷵻�NULL */
static Position FindSibling(Position Parent, int i)
{
	Position Sibling;
	int Limit;

	Limit = M;

	Sibling = NULL;
	if (i == 0)
	{
		if (Parent->Children[1]->KeyNum < Limit)
			Sibling = Parent->Children[1];
	}
	else if (Parent->Children[i - 1]->KeyNum < Limit)
		Sibling = Parent->Children[i - 1];
	else if (i + 1 < Parent->KeyNum && Parent->Children[i + 1]->KeyNum < Limit)
	{
		Sibling = Parent->Children[i + 1];
	}

	return Sibling;
}

/* �����ֵܽڵ㣬��ؼ���������M/2 ;û�з���NULL*/
static Position FindSiblingKeyNum_M_2(Position Parent, int i, int *j)
{
	int Limit;
	Position Sibling;
	Sibling = NULL;

	Limit = LIMIT_M_2;

	if (i == 0)
	{
		if (Parent->Children[1]->KeyNum > Limit)
		{
			Sibling = Parent->Children[1];
			*j = 1;
		}
	}
	else
	{
		if (Parent->Children[i - 1]->KeyNum > Limit)
		{
			Sibling = Parent->Children[i - 1];
			*j = i - 1;
		}
		else if (i + 1 < Parent->KeyNum && Parent->Children[i + 1]->KeyNum > Limit)
		{
			Sibling = Parent->Children[i + 1];
			*j = i + 1;
		}
	}
	return Sibling;
}

/* ��Ҫ��X����Key��ʱ��i��X��Parent��λ�ã�j��KeyҪ�����λ��
   ��Ҫ��Parent����X�ڵ��ʱ��i��Ҫ�����λ�ã�Key��j��ֵû����
 */
static Position InsertElement(int isKey, Position Parent, Position X, KeyType Key, int i, int j)
{

	int k;
	if (isKey)
	{
		/* ����key */
		k = X->KeyNum - 1;
		while (k >= j)
		{
			X->Key[k + 1] = X->Key[k];
			k--;
		}
		X->Key[j] = Key;
		if (Parent != NULL)
		{
			// cout << "PKa:" << Parent->KeyNum <<" i: "<<i<< endl;
			Parent->Key[i] = X->Key[0];
			// cout << "PKa:" << Parent->KeyNum << endl;
		}
		X->KeyNum++;
	}
	else
	{
		/* ����ڵ� */

		/* ����Ҷ�ڵ�������� */
		if (X->Children[0] == NULL)
		{
			if (i > 0)
				Parent->Children[i - 1]->Next = X;
			X->Next = Parent->Children[i];
		}

		k = Parent->KeyNum - 1;
		while (k >= i)
		{
			Parent->Children[k + 1] = Parent->Children[k];
			Parent->Key[k + 1] = Parent->Key[k];
			k--;
		}
		Parent->Key[i] = X->Key[0];
		Parent->Children[i] = X;

		Parent->KeyNum++;
	}
	return X;
}

static Position RemoveElement(int isKey, Position Parent, Position X, int i, int j)
{

	int k, Limit;

	if (isKey)
	{
		Limit = X->KeyNum;
		/* ɾ��key */
		k = j + 1;
		while (k < Limit)
		{
			X->Key[k - 1] = X->Key[k];
			k++;
		}

		X->Key[X->KeyNum - 1] = Unavailable;

		Parent->Key[i] = X->Key[0];

		X->KeyNum--;
	}
	else
	{
		/* ɾ���ڵ� */

		/* �޸���Ҷ�ڵ������ */
		if (X->Children[0] == NULL && i > 0)
		{
			Parent->Children[i - 1]->Next = Parent->Children[i + 1];
		}
		Limit = Parent->KeyNum;
		k = i + 1;
		while (k < Limit)
		{
			Parent->Children[k - 1] = Parent->Children[k];
			Parent->Key[k - 1] = Parent->Key[k];
			k++;
		}

		Parent->Children[Parent->KeyNum - 1] = NULL;
		Parent->Key[Parent->KeyNum - 1] = Unavailable;

		Parent->KeyNum--;
	}
	return X;
}

/* Src��Dst���������ڵĽڵ㣬i��Src��Parent�е�λ�ã�
 ��Src��Ԫ���ƶ���Dst�� ,n���ƶ�Ԫ�صĸ���*/
static Position MoveElement(Position Src, Position Dst, Position Parent, int i, int n)
{
	KeyType TmpKey;
	Position Child;
	int j, SrcInFront;

	SrcInFront = 0;

	if ((unsigned)Src->Key[0]->z32 < (unsigned)Dst->Key[0]->z32)
	{
		SrcInFront = 1;
	}
	// cout<<"Infront:"<<SrcInFront<<endl;
	j = 0;
	/* �ڵ�Src��Dstǰ�� */
	if (SrcInFront)
	{
		if (Src->Children[0] != NULL)
		{
			while (j < n)
			{
				Child = Src->Children[Src->KeyNum - 1];
				RemoveElement(0, Src, Child, Src->KeyNum - 1, 0);
				InsertElement(0, Dst, Child, Unavailable, 0, 0);
				j++;
			}
		}
		else
		{
			while (j < n)
			{
				TmpKey = Src->Key[Src->KeyNum - 1];
				RemoveElement(1, Parent, Src, i, Src->KeyNum - 1);
				InsertElement(1, Parent, Dst, TmpKey, i + 1, 0);
				j++;
			}
		}

		Parent->Key[i + 1] = Dst->Key[0];
		/* ����Ҷ�ڵ��������� */
		if (Src->KeyNum > 0)
			FindMostRight(Src)->Next = FindMostLeft(Dst);
	}
	else
	{
		// cout << "PK:" << Parent->KeyNum << endl;
		if (Src->Children[0] != NULL)
		{
			while (j < n)
			{
				Child = Src->Children[0];
				RemoveElement(0, Src, Child, 0, 0);
				InsertElement(0, Dst, Child, Unavailable, Dst->KeyNum, 0);
				j++;
			}
		}
		else
		{
			while (j < n)
			{
				TmpKey = Src->Key[0];
				// cout << "Src:" << Src << " Dst: " << Dst << endl;

				// cout << "PK1:" << Parent->KeyNum << endl;
				RemoveElement(1, Parent, Src, i, 0);
				// cout << "PK2:" << Parent->KeyNum << endl;
				InsertElement(1, Parent, Dst, TmpKey, i - 1, Dst->KeyNum);
				// cout << "PK3:" << Parent->KeyNum << endl;
				j++;
			}
		}
		// cout << "PK:" << Parent->KeyNum << endl;
		Parent->Key[i] = Src->Key[0];
		if (Src->KeyNum > 0)
			FindMostRight(Dst)->Next = FindMostLeft(Src);
	}

	return Parent;
}

static BPlusTree SplitNode(Position Parent, Position X, int i)
{
	int j, k, Limit;
	Position NewNode;

	NewNode = MallocNewNode();

	k = 0;
	j = X->KeyNum / 2;
	Limit = X->KeyNum;
	// cout << "split detail:" << j << " " << Limit << " parent" << Parent << endl;
	while (j < Limit)
	{
		if (X->Children[0] != NULL)
		{
			NewNode->Children[k] = X->Children[j];
			X->Children[j] = NULL;
		}
		NewNode->Key[k] = X->Key[j];
		X->Key[j] = Unavailable;
		NewNode->KeyNum++;
		X->KeyNum--;
		j++;
		k++;
	}

	if (Parent != NULL)
		InsertElement(0, Parent, NewNode, Unavailable, i + 1, 0);
	else
	{
		/* �����X�Ǹ�����ô�����µĸ������� */
		Parent = MallocNewNode();
		InsertElement(0, Parent, X, Unavailable, 0, 0);
		InsertElement(0, Parent, NewNode, Unavailable, 1, 0);

		return Parent;
	}

	return X;
}

/* �ϲ��ڵ�,X����M/2�ؼ��֣�S�д��ڻ����M/2���ؼ���*/
static Position MergeNode(Position Parent, Position X, Position S, int i)
{
	int Limit;

	/* S�Ĺؼ�����Ŀ����M/2 */
	if (S->KeyNum > LIMIT_M_2)
	{
		/* ��S���ƶ�һ��Ԫ�ص�X�� */
		MoveElement(S, X, Parent, i, 1);
	}
	else
	{
		/* ��Xȫ��Ԫ���ƶ���S�У�����Xɾ�� */
		Limit = X->KeyNum;
		MoveElement(X, S, Parent, i, Limit);
		RemoveElement(0, Parent, X, i, 0);

		free(X);
		X = NULL;
	}

	return Parent;
}
/* (�Ķ�)��Ҫ��X����Key��ʱ��i��X��Parent��λ�ã�j��KeyҪ�����λ��
   ��Ҫ��Parent����X�ڵ��ʱ��i��Ҫ�����λ�ã�Key��j��ֵû����
 */
static Position InsertBulk(int isKey, Position Parent, Position X, KeyType Key, int i, int j)
{

	int k;
	if (isKey)
	{
		/* ����key */
		k = X->KeyNum - 1;
		while (k >= j)
		{
			X->Key[k + 1] = X->Key[k];
			k--;
		}
		X->Key[j] = Key;
		if (Parent != NULL)
		{
			// cout << "PKa:" << Parent->KeyNum <<" i: "<<i<< endl;
			Parent->Key[i] = X->Key[0];
			// cout << "PKa:" << Parent->KeyNum << endl;
		}
		X->KeyNum++;
	}
	else
	{
		/* ����ڵ� */

		/* ����Ҷ�ڵ�������� */
		if (X->Children[0] == NULL)
		{
			if (i > 0)
				Parent->Children[i - 1]->Next = X;
			X->Next = Parent->Children[i];
		}

		k = Parent->KeyNum - 1;
		while (k >= i)
		{
			Parent->Children[k + 1] = Parent->Children[k];
			Parent->Key[k + 1] = Parent->Key[k];
			k--;
		}
		Parent->Key[i] = X->Key[0];
		Parent->Children[i] = X;

		Parent->KeyNum++;
	}
	return X;
}

static BPlusTree RecursiveInsert(BPlusTree T, KeyType Key, int i, BPlusTree Parent)
{
	int j, Limit;
	Position Sibling;
	/* ���ҷ�֧ */
	j = 0;
	while (j < T->KeyNum && (unsigned)Key->z32 >= (unsigned)T->Key[j]->z32)
	{
		// if (Key->z32 == T->Key[j]->z32)
		//     return T;
		// cout<<(unsigned)Key->z32<<" "<<(unsigned)T->Key[j]->z32<<endl;
		j++;
		// if(T->Key[j]==NULL){
		//     break;
		// }
	}
	// cout << "Insert Middle" << endl;
	if (j != 0 && T->Children[0] != NULL)
	{
		j--;
	}
	/* ��Ҷ */
	if (T->Children[0] == NULL)
	{

		T = InsertElement(1, Parent, T, Key, i, j);
		// cout << T->KeyNum << endl;
	}
	/* �ڲ��ڵ� */
	else
	{
		T->Children[j] = RecursiveInsert(T->Children[j], Key, j, T);
		// cout << T->KeyNum << endl;
	}
	// cout << T->KeyNum << endl;
	/* �����ڵ� */

	Limit = M;

	if (T->KeyNum > Limit)
	{
		/* �� */
		if (Parent == NULL)
		{
			/* ���ѽڵ� */
			// cout << "Spliting: " << T->KeyNum << endl;
			T = SplitNode(Parent, T, i);
		}
		else
		{
			// cout << "Trans2slib" << endl;
			Sibling = FindSibling(Parent, i);
			// cout << "slib " << Sibling << "Par" << Parent->KeyNum << endl;
			if (Sibling != NULL)
			{
				/* ��T��һ��Ԫ�أ�Key����Child���ƶ���Sibing�� */
				MoveElement(T, Sibling, Parent, i, 1);
			}
			else
			{
				/* ���ѽڵ� */
				T = SplitNode(Parent, T, i);
			}
		}
	}
	if (Parent != NULL)
	{
		Parent->Key[i] = T->Key[0];
		// cout << "Par " << Parent->KeyNum << endl;
	}

	// cout << "Insert Last:" << T->KeyNum << endl;
	return T;
}
/* ���� */
extern BPlusTree Insert(BPlusTree T, KeyType Key)
{
	return RecursiveInsert(T, Key, 0, NULL);
}
BPlusTree bulkInit(vector<ZTuple *> &VZ)
{

	int outflow = 0;
	BPlusTree T = MallocNewNode();
	if (VZ.size() <= M)
	{
		for (int i = 0; i < VZ.size(); i++)
		{
			T->Key[i] = VZ[i];
		}
		T->KeyNum = VZ.size();
		T->Next = NULL;
		return T;
	}
	else
	{
		outflow = 1;
		// ��������
		BPlusNode *childHead = T;
		for (int i = 0; i < VZ.size(); i++)
		{
			if (T->KeyNum < M)
			{
				T->Key[T->KeyNum] = VZ[i];
				T->KeyNum++;
			}
			else
			{
				BPlusNode *nextNode = MallocNewNode();
				T->Next = nextNode;
				T = nextNode;
				T->Key[T->KeyNum] = VZ[i];
				T->KeyNum++;
			}
		}
		BPlusNode *fatherhead = NULL;
		while (outflow)
		{
			// ���츸����
			T = MallocNewNode();
			fatherhead = T;
			outflow = 0;
			int debugv1 = 0, debugv2 = 0;
			for (BPlusNode *cptr = childHead; cptr != NULL; cptr = cptr->Next)
			{
				debugv1++;
				if (T->KeyNum < M)
				{
					T->Key[T->KeyNum] = cptr->Key[0];
					T->Children[T->KeyNum] = cptr;
					T->KeyNum++;
				}
				else
				{
					outflow = 1;
					BPlusNode *nextNode = MallocNewNode();
					debugv2++;
					T->Next = nextNode;
					T = nextNode;
					T->Key[T->KeyNum] = cptr->Key[0];
					T->Children[T->KeyNum] = cptr;
					T->KeyNum++;
				}
			}
			// cout << "ckpt arrived" <<" Num leafs:"<<debugv1<<" "<<" L3: "<<debugv2<< endl;
			// exit(1);
			// ���±���ά��
			childHead = fatherhead;
		}
		return fatherhead;
		// cout<<"I'm free!"<<endl;
		// cout<<fatherhead->KeyNum<<endl;
	}
	return NULL;
}
BPlusNode *fastBulkMerge(BPlusNode *curptr, vector<ZTuple *> &VZ)
{
	cout << "in FastBM" << endl;
	BPlusNode *oldhead = curptr;
	BPlusNode *newhead = MallocNewNode();
	BPlusNode *workingNode = newhead;
	int curidx = 0;
	int vzidx = 0;
	int skipNum = 0;
	while (curptr != NULL && vzidx < VZ.size())
	{

		KeyType vzkey = VZ[vzidx];
		KeyType oldBTKey = curptr->Key[curidx];

		// if (vzidx >= 615000)
		// {
		//     cout  << vzidx << " " << curidx << " KN<" << curptr->KeyNum<<"> "<<endl;
		//     cout<< ((unsigned)vzkey->z32 )<<" "<<oldBTKey<<" "<<((unsigned)oldBTKey->z32)<<endl;
		// }
		if (curidx == 0 && ((unsigned)vzkey->z32 > (unsigned)curptr->Key[curptr->KeyNum - 1]->z32))
		{
			// skip
			skipNum++;
			if (workingNode->KeyNum == 0)
			{
				workingNode->KeyNum = curptr->KeyNum;
				workingNode->Key = curptr->Key;
				workingNode->Children = curptr->Children;
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				curptr = curptr->Next;
			}
			else
			{
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				workingNode->KeyNum = curptr->KeyNum;
				workingNode->Key = curptr->Key;
				workingNode->Children = curptr->Children;
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
			}
			continue;
		}

		if ((unsigned)vzkey->z32 < (unsigned)oldBTKey->z32)
		{

			if (workingNode->KeyNum < M)
			{
				workingNode->Key[workingNode->KeyNum] = vzkey;
				workingNode->KeyNum++;
			}
			else
			{
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				workingNode->Key[workingNode->KeyNum] = vzkey;
				workingNode->KeyNum++;
			}
			vzidx += 1;
		}
		else
		{
			if (workingNode->KeyNum < M)
			{
				workingNode->Key[workingNode->KeyNum] = oldBTKey;
				workingNode->KeyNum++;
			}
			else
			{
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				workingNode->Key[workingNode->KeyNum] = oldBTKey;
				workingNode->KeyNum++;
			}
			curidx++;
			if (curidx >= curptr->KeyNum)
			{
				// cout << '\r' << vzidx << " " << curidx << " " << curptr->KeyNum<<" "<<curptr->Next;
				// if ( curptr->Next!=NULL){
				//     cout<< " KN:"<<curptr->Next->KeyNum<<"<<              ";
				// }
				curidx = 0;
				curptr = curptr->Next;
			}
		}
		// if (vzidx >= 615000)
		// {
		//     cout <<  vzidx << " " << curidx << " CP<" << curptr<<"> "<<endl;
		// }
	}
	// cout << "half way" << endl;
	cout << "Skiped Number:" << skipNum << endl;
	if ((vzidx < VZ.size()))
	{
		cout << "bRANCH1" << endl;
	}
	else
	{
		cout << "bRANCH2" << endl;
	}
	while (vzidx < VZ.size())
	{
		KeyType vzkey = VZ[vzidx];
		if (workingNode->KeyNum < M)
		{
			workingNode->Key[workingNode->KeyNum] = vzkey;
			workingNode->KeyNum++;
		}
		else
		{
			workingNode->Next = MallocNewNode();
			workingNode = workingNode->Next;
			workingNode->Key[workingNode->KeyNum] = vzkey;
			workingNode->KeyNum++;
		}
		vzidx += 1;
	}
	while (curptr != NULL)
	{
		KeyType oldBTKey = curptr->Key[curidx];
		if (oldBTKey == NULL)
		{
			break;
		}
		if (workingNode->KeyNum < M)
		{
			workingNode->Key[workingNode->KeyNum] = oldBTKey;
			workingNode->KeyNum++;
		}
		else
		{
			workingNode->Next = MallocNewNode();
			workingNode = workingNode->Next;
			workingNode->Key[workingNode->KeyNum] = oldBTKey;
			workingNode->KeyNum++;
		}

		if (curidx < curptr->KeyNum)
		{
			curidx++;
		}
		else
		{
			curidx = 0;
			curptr = curptr->Next;
		}
	}
	workingNode = oldhead;
	// cout << "deleting" << endl;
	// while (workingNode != NULL)
	// {
	//     BPlusNode *pdelete;
	//     workingNode = workingNode->Next;
	//     delete pdelete;
	//     /* code */
	// }
	return newhead;
}

BPlusNode *bulkMerge(BPlusNode *curptr, vector<ZTuple *> &VZ)
{
	cout << "in BM" << endl;
	BPlusNode *oldhead = curptr;
	BPlusNode *newhead = MallocNewNode();
	BPlusNode *workingNode = newhead;
	int curidx = 0;
	int vzidx = 0;
	while (curptr != NULL && vzidx < VZ.size())
	{

		KeyType vzkey = VZ[vzidx];
		KeyType oldBTKey = curptr->Key[curidx];
		// if (vzidx >= 615000)
		// {
		//     cout  << vzidx << " " << curidx << " KN<" << curptr->KeyNum<<"> "<<endl;
		//     cout<< ((unsigned)vzkey->z32 )<<" "<<oldBTKey<<" "<<((unsigned)oldBTKey->z32)<<endl;
		// }
		if ((unsigned)vzkey->z32 < (unsigned)oldBTKey->z32)
		{

			if (workingNode->KeyNum < M)
			{
				workingNode->Key[workingNode->KeyNum] = vzkey;
				workingNode->KeyNum++;
			}
			else
			{
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				workingNode->Key[workingNode->KeyNum] = vzkey;
				workingNode->KeyNum++;
			}
			vzidx += 1;
		}
		else
		{
			if (workingNode->KeyNum < M)
			{
				workingNode->Key[workingNode->KeyNum] = oldBTKey;
				workingNode->KeyNum++;
			}
			else
			{
				workingNode->Next = MallocNewNode();
				workingNode = workingNode->Next;
				workingNode->Key[workingNode->KeyNum] = oldBTKey;
				workingNode->KeyNum++;
			}
			curidx++;
			if (curidx >= curptr->KeyNum)
			{
				// cout << '\r' << vzidx << " " << curidx << " " << curptr->KeyNum<<" "<<curptr->Next;
				// if ( curptr->Next!=NULL){
				//     cout<< " KN:"<<curptr->Next->KeyNum<<"<<              ";
				// }
				curidx = 0;
				curptr = curptr->Next;
			}
		}
		// if (vzidx >= 615000)
		// {
		//     cout <<  vzidx << " " << curidx << " CP<" << curptr<<"> "<<endl;
		// }
	}
	cout << "half way" << endl;
	if ((vzidx < VZ.size()))
	{
		cout << "bRANCH1" << endl;
	}
	else
	{
		cout << "bRANCH2" << endl;
	}
	while (vzidx < VZ.size())
	{
		KeyType vzkey = VZ[vzidx];
		if (workingNode->KeyNum < M)
		{
			workingNode->Key[workingNode->KeyNum] = vzkey;
			workingNode->KeyNum++;
		}
		else
		{
			workingNode->Next = MallocNewNode();
			workingNode = workingNode->Next;
			workingNode->Key[workingNode->KeyNum] = vzkey;
			workingNode->KeyNum++;
		}
		vzidx += 1;
	}
	while (curptr != NULL)
	{
		KeyType oldBTKey = curptr->Key[curidx];
		if (oldBTKey == NULL)
		{
			break;
		}
		if (workingNode->KeyNum < M)
		{
			workingNode->Key[workingNode->KeyNum] = oldBTKey;
			workingNode->KeyNum++;
		}
		else
		{
			workingNode->Next = MallocNewNode();
			workingNode = workingNode->Next;
			workingNode->Key[workingNode->KeyNum] = oldBTKey;
			workingNode->KeyNum++;
		}

		if (curidx < curptr->KeyNum)
		{
			curidx++;
		}
		else
		{
			curidx = 0;
			curptr = curptr->Next;
		}
	}
	workingNode = oldhead;
	// cout << "deleting" << endl;
	// while (workingNode != NULL)
	// {
	//     BPlusNode *pdelete;
	//     workingNode = workingNode->Next;
	//     delete pdelete;
	//     /* code */
	// }
	return newhead;
}
// CardIndex *C;
CardIndex *LinkedList2CardIndex(BPlusNode *Head, MADENet *Net)
{
	CardIndex *C = new CardIndex;
	C->Head = Head;
	BPlusNode *fatherhead;
	BPlusNode *childHead = Head;
	C->Mnet = Net;
	int curLinkTopLen = 0;
	while (true) // ����������������������������
	{
		curLinkTopLen = 0;
		BPlusNode *curptr = childHead;

		// �Դ����Ĵ˲����ά������function
		maintainBPlusProperty(curptr);

		while (curptr != NULL)
		{
			curLinkTopLen += 1;
			curptr = curptr->Next;
		}
		cout << "curlen: " << curLinkTopLen << endl;

		if (curLinkTopLen < MAXL1CHILD)
		{
			break;
		}
		// ������Ƚ�������
		// ���츸����
		BPlusNode *T = MallocNewNode();
		fatherhead = T;
		for (BPlusNode *cptr = childHead; cptr != NULL; cptr = cptr->Next)
		{
			if (T->KeyNum < M)
			{
				T->Key[T->KeyNum] = cptr->Key[0];
				T->Children[T->KeyNum] = cptr;
				T->KeyNum++;
			}
			else
			{
				BPlusNode *nextNode = MallocNewNode();
				T->Next = nextNode;
				T = nextNode;
				T->Key[T->KeyNum] = cptr->Key[0];
				T->Children[T->KeyNum] = cptr;
				T->KeyNum++;
			}
		}
		// ���±���ά��
		childHead = fatherhead;
	}
	maintainBPlusProperty(childHead);
	int i = -1;
	C->trans = new MiddleLayer;
	for (int ix = 0; ix < MAXL1CHILD; ix++)
	{
		C->trans->transferLayer[ix] = NULL;
		C->trans->Flag[ix] = 0;
	}
	// cout<<"starting 2 make somethnig new"<<endl;

	for (auto curptr = childHead; curptr != NULL; curptr = curptr->Next)
	{
		i++;
		float cdf0 = cdfCalculate(C->Mnet, curptr->Key[0]);
		int belong = int(cdf0 * MAXL1CHILD);
		// cout << "Hashing " << belong << endl;
		if (belong < 0)
		{
			belong = 0;
		}
		if (belong >= MAXL1CHILD)
		{
			belong = MAXL1CHILD - 1;
		}
		if (C->trans->transferLayer[belong] == NULL)
		{
			C->trans->transferLayer[belong] = curptr;
		}
		else
		{
			BPlusNode *BPN = C->trans->transferLayer[belong];
			if (BPN->Next == (BPlusNode *)-1)
			{ // ��2�����ϵĹ�ϣ��ͻ,extend,��ʱ���������˵����
				BPN->Key[BPN->KeyNum] = curptr->Key[0];
				BPN->Children[BPN->KeyNum] = curptr;
				BPN->KeyNum++;
				maintainLinearNode(BPN);
			}
			else
			{ // ��һ�ι�ϣ��ͻ�������ĸ߶�����һ��
				// cout<<"First Hash conflit"<<endl;
				BPlusNode *ReplaceN = MallocNewNode();
				ReplaceN->Next = (BPlusNode *)-1;
				ReplaceN->Key[0] = BPN->Key[0];
				ReplaceN->Children[0] = BPN;
				ReplaceN->Key[1] = curptr->Key[0];
				ReplaceN->Children[1] = curptr;
				ReplaceN->KeyNum = 2;
				C->trans->transferLayer[belong] = ReplaceN;
				maintainLinearNode(ReplaceN);
			}
		}
		// cout<<i<<"Bulk CDF: "<<cdf<<" IF CDF HASH:" << cdf * curLinkTopLen <<endl;
	}
	BPlusNode *firstNZ = NULL;
	BPlusNode *preSet = NULL;
	for (int ix = 0; ix < MAXL1CHILD; ix++)
	{

		if (C->trans->transferLayer[ix] == NULL)
		{
			// cout << "IX" << ix << " " << NULL << endl;
			C->trans->transferLayer[ix] = firstNZ;
		}
		else
		{
			if (C->trans->transferLayer[ix] != NULL)
			{
				// ��һ�����ڵĻ����split��,������ӱ��
				C->trans->Flag[ix] = 1;
				if (firstNZ == NULL)
				{
					preSet = C->trans->transferLayer[ix];
				}
				firstNZ = C->trans->transferLayer[ix];
			}
		}
	}
	for (int ix = 0; ix < MAXL1CHILD; ix++)
	{
		if (C->trans->transferLayer[ix] == NULL)
		{
			C->trans->transferLayer[ix] = preSet;
		}
		else
		{
			break;
		}
	}
	C->trans->curnum = MAXL1CHILD;
	// exit(1);
	// cout << "reting" << endl;
	return C;
}
void maintainLinearNode(BPlusNode *ptr)
{
	// cout<<"Begin Maintaining"<<endl;
	ptr->Zmin = ptr->Key[0]->z32;
	ptr->Zmax = ptr->Key[ptr->KeyNum - 1]->z32;
	/*cout << "Info: Keynum:" << ptr->KeyNum << " zmin" << ptr->Zmin << " zmax" << ptr->Zmax << endl;
	cout << "Is min<max:" << (ptr->Zmin < ptr->Zmax )<< endl;*/
	if (ptr->Zmax == ptr->Zmin)
	{
		return; // no need to maintain if all are equal;
	}
	double err = M / 10.0;
	double k_up = FLT_MAX;
	double k_down = -FLT_MAX;
	ptr->linNum = 0;
	double x0 = (ptr->Key[0]->z32 - ptr->Zmin) / (0.0 + ptr->Zmax - ptr->Zmin);
	double y0 = 0;
	for (int i = 1; i < ptr->KeyNum; i++)
	{
		double delta_y = (i - y0 + 0.0);
		double delta_x = (((ptr->Key[i]->z32 - ptr->Zmin) / (0.0 + ptr->Zmax - ptr->Zmin)) - x0 + 0.0);
		// cout << i <<" " << x0 << " " << delta_x << endl;
		if (delta_x == 0)
		{
			continue;
		}
		if ((k_up * delta_x > (delta_y + err)) && (k_down * delta_x < (delta_y - err)))
		{
			// inside the cone,shink
			k_up = (delta_y + err) / (0.000001 + delta_x);
			k_down = (delta_y - err) / (0.000001 + delta_x);
		}
		else
		{
			ptr->LinearX[ptr->linNum].kup = k_up;
			ptr->LinearX[ptr->linNum].kdown = k_down;
			ptr->LinearX[ptr->linNum].x0 = x0;
			ptr->LinearX[ptr->linNum].y0 = y0;
			x0 = ((ptr->Key[i]->z32 - ptr->Zmin) / (0.0 + ptr->Zmax - ptr->Zmin));
			y0 = i;
			k_up = FLT_MAX;
			k_down = -FLT_MAX;
			ptr->linNum++;
		}
	}
	/*for(int i=0;i<ptr->linNum;i++){
		cout<<ptr->LinearX[i].x0<<" "<<ptr->LinearX[i].y0<<" "<<ptr->LinearX[i].kdown<<" "<<ptr->LinearX[i].kup<<endl;
	}*/
	/*cout<<"End of Maintain"<<endl;
	cout<<ptr->linNum<<endl;
	exit(1);*/
	if (ptr->linNum == 0)
	{
		ptr->LinearX[ptr->linNum].kup = k_up;
		ptr->LinearX[ptr->linNum].kdown = k_down;
		ptr->LinearX[ptr->linNum].x0 = x0;
		ptr->LinearX[ptr->linNum].y0 = y0;
		ptr->linNum++;
	}
}
void maintainBPlusProperty(BPlusNode *Head)
{
	int pre = 0;
	for (BPlusNode *ptr = Head; ptr != NULL; ptr = ptr->Next)
	{
		ptr->preNum = pre;
		pre += ptr->KeyNum;
		maintainLinearNode(ptr);
		// maintain linear stuff
	}
}

BPlusNode *MergeLinkedList(BPlusNode *L1, ZTab *ZT)
{
	vector<ZTuple *> VZ = ZT->D;
	// sort(VZ.begin(), VZ.end(), ZTcmp);
	if (L1 == NULL)
	{
		BPlusNode *head;
		cout << "Cold Start" << endl;
		BPlusNode *T = MallocNewNode();
		head = T;
		int pre = 0;
		for (int i = 0; i < VZ.size(); i++)
		{
			if (T->KeyNum < M)
			{
				T->Key[T->KeyNum] = VZ[i];
				T->KeyNum++;
			}
			else
			{
				T->preNum = pre;
				pre = i;
				BPlusNode *nextNode = MallocNewNode();
				T->Next = nextNode;
				T = nextNode;
				T->Key[T->KeyNum] = VZ[i];
				T->KeyNum++;
			}
		}
		// typedef std::chrono::high_resolution_clock Clock;
		// auto t3 = Clock::now(); // ��ʱ��ʼ
		// �ںϲ�������,ά����ײ�ڵ����Ϣ
		// maintainBPlusProperty(head);
		// auto t4 = Clock::now(); // ��ʱ��ʼ
		// cout << "Maintain Time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count()) / (ZT->r + 0.0) << endl;
		return head;
	}
	else
	{
		BPlusNode *newLinklist = bulkMerge(L1, VZ);
		// maintainBPlusProperty(newLinklist);
		return newLinklist;
	}
}
BPlusTree bulkInsert(BPlusTree T, ZTab *ZT)
{
	// cout<<"Hi"<<endl;
	vector<ZTuple *> VZ = ZT->D;
	sort(VZ.begin(), VZ.end(), ZTcmp);
	if (T == NULL)
	{
		cout << "Cold Start" << endl;
		T = bulkInit(VZ);
		return T;
		/* code */
	}
	else
	{
		BPlusNode *curptr = T;
		while (curptr->Children[0] != NULL)
		{
			curptr = curptr->Children[0];
		}
		// MERGE
		typedef std::chrono::high_resolution_clock Clock;
		auto t3 = Clock::now(); // ��ʱ��ʼ
		BPlusNode *newLinklist = bulkMerge(curptr, VZ);

		auto t4 = Clock::now(); // ��ʱ��ʼ
		cout << "Merge Time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count()) / (ZT->r + 0.0) << endl;

		cout << "Bulk merged" << endl;
		BPlusNode *childHead = newLinklist;
		BPlusNode *fatherhead;
		int outflow = 1;
		while (outflow)
		{
			// ���츸����
			T = MallocNewNode();
			fatherhead = T;
			outflow = 0;
			int debugv1 = 0, debugv2 = 0;
			for (BPlusNode *cptr = childHead; cptr != NULL; cptr = cptr->Next)
			{
				debugv1++;
				if (T->KeyNum < M)
				{
					T->Key[T->KeyNum] = cptr->Key[0];
					T->Children[T->KeyNum] = cptr;
					T->KeyNum++;
				}
				else
				{
					outflow = 1;
					BPlusNode *nextNode = MallocNewNode();
					debugv2++;
					T->Next = nextNode;
					T = nextNode;
					T->Key[T->KeyNum] = cptr->Key[0];
					T->Children[T->KeyNum] = cptr;
					T->KeyNum++;
				}
			}

			// cout << "ckpt arrived" <<" Num leafs:"<<debugv1<<" "<<" L3: "<<debugv2<< endl;
			// exit(1);
			// ���±���ά��
			childHead = fatherhead;
		}
		// auto tx = Clock::now();
		// int inc=0;
		// for (auto ptrx = fatherhead->Children[0]; ptrx != NULL; ptrx = ptrx->Next)
		// {
		//     inc++;
		//     int belong = getBelongNum(C, ptrx->Key[0]);
		// }
		// cout<<inc<<endl;
		// auto t6 = Clock::now(); // ��ʱ��ʼ
		// cout << "CDF Time:" << (std::chrono::duration_cast<std::chrono::nanoseconds>(tx - t6).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(tx - t6).count()) / (ZT->r + 0.0) << endl;
		auto t5 = Clock::now(); // ��ʱ��ʼ
		cout << "BT MAKE Time:" << (std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count()) / (ZT->r + 0.0) << endl;
		return fatherhead;
	}
	return NULL;
}
static BPlusTree RecursiveRemove(BPlusTree T, KeyType Key, int i, BPlusTree Parent)
{
	cout << "shit in" << endl;
	exit(1);
	int j, NeedAdjust;
	Position Sibling, Tmp;

	Sibling = NULL;

	/* ���ҷ�֧ */
	j = 0;
	while (j < T->KeyNum && (unsigned)Key->z32 >= (unsigned)T->Key[j]->z32)
	{
		if (Key == T->Key[j])
			break;
		j++;
		if (T->Key[j] == NULL)
		{
			break;
		}
	}

	if (T->Children[0] == NULL)
	{
		/* û�ҵ� */
		if (Key->z32 != T->Key[j]->z32 || j == T->KeyNum)
			return T;
	}
	else if (j == T->KeyNum || Key->z32 < T->Key[j]->z32)
		j--;

	/* ��Ҷ */
	if (T->Children[0] == NULL)
	{
		T = RemoveElement(1, Parent, T, i, j);
	}
	else
	{
		T->Children[j] = RecursiveRemove(T->Children[j], Key, j, T);
	}

	NeedAdjust = 0;
	/* ���ĸ�������һƬ��Ҷ���������������2��M֮�� */
	if (Parent == NULL && T->Children[0] != NULL && T->KeyNum < 2)
		NeedAdjust = 1;
	/* �����⣬���з���Ҷ�ڵ�Ķ�������[M/2]��M֮�䡣(����[]��ʾ����ȡ��) */
	else if (Parent != NULL && T->Children[0] != NULL && T->KeyNum < LIMIT_M_2)
		NeedAdjust = 1;
	/* ���Ǹ�����Ҷ�йؼ��ֵĸ���Ҳ��[M/2]��M֮�� */
	else if (Parent != NULL && T->Children[0] == NULL && T->KeyNum < LIMIT_M_2)
		NeedAdjust = 1;

	/* �����ڵ� */
	if (NeedAdjust)
	{
		/* �� */
		if (Parent == NULL)
		{
			if (T->Children[0] != NULL && T->KeyNum < 2)
			{
				Tmp = T;
				T = T->Children[0];
				free(Tmp);
				return T;
			}
		}
		else
		{
			/* �����ֵܽڵ㣬��ؼ�����Ŀ����M/2 */
			Sibling = FindSiblingKeyNum_M_2(Parent, i, &j);
			if (Sibling != NULL)
			{
				MoveElement(Sibling, T, Parent, j, 1);
			}
			else
			{
				if (i == 0)
					Sibling = Parent->Children[1];
				else
					Sibling = Parent->Children[i - 1];

				Parent = MergeNode(Parent, T, Sibling, i);
				T = Parent->Children[i];
			}
		}
	}

	return T;
}

// /* ɾ�� */
// extern BPlusTree Remove(BPlusTree T,KeyType Key){
//     return RecursiveRemove(T, Key, 0, NULL);
// }

/* ���� */
extern BPlusTree Destroy(BPlusTree T)
{
	int i, j;
	if (T != NULL)
	{
		i = 0;
		while (i < T->KeyNum + 1)
		{
			Destroy(T->Children[i]);
			i++;
		}

		// printf("Destroy:(");
		j = 0;
		// while (j < T->KeyNum) /*  T->Key[i] != Unavailable*/
		//     printf("%d:", T->Key[j++]);
		// printf(") ");
		free(T);
		T = NULL;
	}

	return T;
}

int recnum = 0;
Position LastNode = NULL;
static void RecursiveTravel(BPlusTree T, int Level)
{
	int i;
	if (T != NULL)
	{
		// printf("  ");
		// printf("[Level:%d]-->", Level);
		// printf("(");
		i = 0;
		// while (i < T->KeyNum) /*  T->Key[i] != Unavailable*/
		//     printf("%d:", T->Key[i++]->z32);
		// printf(")");
		// return;
		Level++;
		if (T->Children[0] == NULL)
		{
			if (LastNode == NULL)
			{
				LastNode = T;
			}
			else
			{
				LastNode->Next = T;
				LastNode = T;
			}
			recnum += T->KeyNum;
		}
		i = 0;
		while (i <= T->KeyNum)
		{
			RecursiveTravel(T->Children[i], Level);
			i++;
		}
	}
}

// int TRAN;
/* �����޸�ָ��˳�� */
extern void Travel(BPlusTree T)
{
	recnum = 0;
	RecursiveTravel(T, 0);
	LastNode->Next = NULL;
	// cout<<"TOLN:"<<recnum<<endl;
	// printf("\n");
}

/* ������Ҷ�ڵ������ */
extern void TravelData(BPlusTree T)
{
	Position Tmp;
	int i;
	if (T == NULL)
		return;
	printf("All Data:");
	Tmp = T;
	while (Tmp->Children[0] != NULL)
		Tmp = Tmp->Children[0];
	/* ��һƬ��Ҷ */
	while (Tmp != NULL)
	{
		i = 0;
		while (i < Tmp->KeyNum)
			printf(" %d", Tmp->Key[i++]);
		Tmp = Tmp->Next;
	}
}

/* ������Ҷ�ڵ������ */
// extern void TravelData(BPlusTree T)
//{
//     Position Tmp;
//     int i;
//     if (T == NULL)
//         return;
//     printf("All Data:");
//     Tmp = T;
//     while (Tmp->Children[0] != NULL)
//         Tmp = Tmp->Children[0];
//     /* ��һƬ��Ҷ */
//     while (Tmp != NULL)
//     {
//         i = 0;
//         while (i < Tmp->KeyNum)
//             printf(" %d", Tmp->Key[i++]);
//         Tmp = Tmp->Next;
//     }
// }
// MADENet* loadMade(string filePath)
//{
//     ifstream infile(filePath);
//     if (!infile.is_open())
//     {
//         cout << "Fail to Net load Tree" << endl;
//         return NULL;
//     }
//     int bittol;
//     // infile >> bittol;
//     // cout<<"bt:"<<bittol<<endl;
//     MADENet* ret = new MADENet;
//     infile >> ret->zdr >> ret->zdc >> ret->connectlen >> ret->leafnums;
//     bittol = ret->zdc;
//     ret->diglen = bittol;
//     ret->fc1w = new float[bittol * bittol];
//     ret->fc2w = new float[bittol * bittol];
//     ret->fc1b = new float[bittol];
//     ret->fc2b = new float[bittol];
//     for (int i = 0; i < bittol; i++)
//     {
//         for (int j = 0; j < bittol; j++)
//         {
//             infile >> ret->fc1w[i * bittol + j];
//         }
//     }
//     for (int i = 0; i < bittol; i++)
//     {
//         infile >> ret->fc1b[i];
//     }
//     for (int i = 0; i < bittol; i++)
//     {
//         for (int j = 0; j < bittol; j++)
//         {
//             infile >> ret->fc2w[i * bittol + j];
//         }
//     }
//     for (int i = 0; i < bittol; i++)
//     {
//         infile >> ret->fc2b[i];
//     }
//     int strcord = 0;
//     infile.close();
//     return ret;
// }
void MadeIndexInfer(bool *xinput, float *out, int preLen, MADENet *net, float *middle)
{
	int winlen = net->connectlen;
	for (int i = 0; i < preLen; i++)
	{
		middle[i] = net->fc1b[i];
		// cout<<middle[i]<<endl;
		for (int j = max(i - winlen, 0); j < i; j++)
		{
			// cout<<"j"<<j<<endl;
			if (j >= i)
			{
				break;
			}
			// cout<<"xij: "<<xinput[j]<<endl;
			middle[i] += (xinput[j] * net->fc1w[i * net->diglen + j]);
		}
		if (middle[i] < 0)
		{
			middle[i] = 0;
		}
	}
	for (int i = 0; i < preLen; i++)
	{
		out[i] = net->fc2b[i];
		for (int j = max(i - winlen, 0); j < i; j++)
		{
			if (j >= i)
			{
				break;
			}
			out[i] += (middle[j] * net->fc2w[i * net->diglen + j]);
		}
		out[i] = (1.0) / (1.0 + exp(-out[i]));
	}
}

float cdfCalculate(MADENet *Mnet, ZTuple *ztup)
{
	float out[50];
	float mid[50];
	MadeIndexInfer(ztup->bin, out, 32, Mnet, mid);
	float cdf = 0;
	float acc = 1.0;
	for (int j = 0; j < 32; j++)
	{
		float onep = out[j];
		cdf += (acc * (1 - onep) * ztup->bin[j]);
		acc *= (onep * ztup->bin[j] + (1 - onep) * (1 - ztup->bin[j]));
	}
	return cdf;
}

CardIndex *InitCardIndex(ZTab *ZT, int L2num)
{
	Unavailable = new ZTuple;
	Unavailable->z32 = INT_MIN;
	CardIndex *C = new CardIndex;
	MADENet *Mnet = loadMade("./Model/MadeRootP0");
	C->Mnet = Mnet;
	string firstFilePath = "./data/ZD0.txt";
	MiddleLayer *MidL = new MiddleLayer;
	C->trans = MidL;
	C->trans->curnum = L2num;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		C->trans->transferLayer[i] = Initialize();
	}
	return C;
	vector<ZTuple *> VZ = ZT->D;
	sort(VZ.begin(), VZ.end(), ZTcmp);
	int u;
	int batchN = 30;
	for (int i = 0; i < (ZT->r) / batchN; i++)
	{
		int ub = min((i + 1) * batchN, ZT->r);
		int lb = i * batchN;
		float cdfl = cdfCalculate(C->Mnet, VZ[lb]);
		float cdfu = cdfCalculate(C->Mnet, VZ[ub]);
		int belongl = cdfl / (1.0 / C->trans->curnum);
		int belongu = cdfu / (1.0 / C->trans->curnum);
		// cout<<belongl<<belongu<<endl;
		if (belongl == belongu)
		{
			for (int j = lb; j < ub; j++)
			{
				C->trans->transferLayer[belongl] = Insert(C->trans->transferLayer[belongl], VZ[j]);
				u = j;
			}
		}
		else
		{
			for (int j = lb; j < ub; j++)
			{
				float cdf0 = cdfCalculate(C->Mnet, VZ[j]);
				int belong = cdf0 / (1.0 / C->trans->curnum);
				int flag = 0;
				C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong], VZ[j]);
				u = j;
			}
		}
		// float cdf0 = cdfCalculate(C->Mnet, VZ[i]);
		// int belong = cdf0 / (1.0 / C->trans->curnum);
		// int flag = 0;
		// C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong],ZT->D[i]);
	}
	u++;
	while (u < ZT->r)
	{
		float cdf0 = cdfCalculate(C->Mnet, VZ[u]);
		int belong = cdf0 / (1.0 / C->trans->curnum);
		int flag = 0;
		C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong], VZ[u]);
		u++;
	}
	cout << u << endl;
	LastNode = NULL;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		BPlusTree tmp = C->trans->transferLayer[i];
		Travel(tmp);
		LastNode = NULL;
	}
	return C;
}

void StaticInsert(CardIndex *C, ZTab *ZT)
{
	vector<ZTuple *> VZ = ZT->D;
	sort(VZ.begin(), VZ.end(), ZTcmp);
	int u;
	int batchN = 30;
	for (int i = 0; i < (ZT->r) / batchN; i++)
	{
		int ub = min((i + 1) * batchN, ZT->r);
		int lb = i * batchN;
		float cdfl = cdfCalculate(C->Mnet, VZ[lb]);
		float cdfu = cdfCalculate(C->Mnet, VZ[ub]);
		int belongl = cdfl / (1.0 / C->trans->curnum);
		int belongu = cdfu / (1.0 / C->trans->curnum);
		// cout<<belongl<<belongu<<endl;
		if (belongl == belongu)
		{
			for (int j = lb; j < ub; j++)
			{
				C->trans->transferLayer[belongl] = Insert(C->trans->transferLayer[belongl], VZ[j]);
				u = j;
			}
		}
		else
		{
			for (int j = lb; j < ub; j++)
			{
				float cdf0 = cdfCalculate(C->Mnet, VZ[j]);
				int belong = cdf0 / (1.0 / C->trans->curnum);
				// int flag = 0;
				C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong], VZ[j]);
				u = j;
			}
		}
		// float cdf0 = cdfCalculate(C->Mnet, VZ[i]);
		// int belong = cdf0 / (1.0 / C->trans->curnum);
		// int flag = 0;
		// C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong],ZT->D[i]);
	}
	u++;

	while (u < ZT->r)
	{
		float cdf0 = cdfCalculate(C->Mnet, VZ[u]);
		int belong = cdf0 / (1.0 / C->trans->curnum);
		// int flag = 0;
		C->trans->transferLayer[belong] = Insert(C->trans->transferLayer[belong], VZ[u]);
		u++;
	}
	// LastNode = NULL;
	// for (int i = 0; i < C->trans->curnum; i++)
	// {
	//     BPlusTree tmp = C->trans->transferLayer[i];
	//     Travel(tmp);
	//     LastNode = NULL;
	// }
}
void CardIndexReport(CardIndex *C)
{
	int cap = 0;
	cout << "\n-----------------\n";
	for (int i = 0; i < C->trans->curnum; i++)
	{
		BPlusTree Ptemp = C->trans->transferLayer[i];
		Position Tmp;
		// cout << "Ci: " << i << " " << Ptemp->KeyNum << endl;
		Tmp = Ptemp;
		while (Tmp->Children[0] != NULL)
			Tmp = Tmp->Children[0];
		int tolL = 0;
		/* ��һInsert */
		// cout<<"hell"<<endl;
		while (Tmp != NULL)
		{
			tolL += Tmp->KeyNum;
			Tmp = Tmp->Next;
		}
		// cout << i << " " << tolL << endl;
		cap += tolL;
	}
	cout << "Toltal records: " << cap << endl;
}

void reBalance(CardIndex *C, string MadeFilePath)
{
	// for (int i = 0; i < C->trans->curnum; i++)
	// {
	//     cout<<C->trans->transferLayer[i]->KeyNum<<" ";
	// }
	// CardIndexReport(C);

	LastNode = NULL;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		BPlusTree tmp = C->trans->transferLayer[i];
		Travel(tmp);
		LastNode = NULL;
	}

	// CardIndexReport(C);
	C->Mnet = NULL;
	cout << "NOTFILE" << endl;
	exit(1);
	int modifiedTrees = 0;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		// cout<<i<<endl;
		if ((C->trans->transferLayer[i]->KeyNum) == 0)
		{
			continue;
		}
		ZTuple *l = C->trans->transferLayer[i]->Key[0];
		ZTuple *u = C->trans->transferLayer[i]->Key[(C->trans->transferLayer[i]->KeyNum) - 1];
		float cdfl = cdfCalculate(C->Mnet, l);
		float cdfu = cdfCalculate(C->Mnet, u);
		int belongl = cdfl / (1.0 / C->trans->curnum);
		int belongu = cdfu / (1.0 / C->trans->curnum);
		// cout << "SubT:" << i << " " << belongl << " " << belongu << endl;
		if (i != belongl || i != belongu)
		{
			// cout << "Imbalance " << i << endl;
			modifiedTrees++;
			BPlusTree Ptemp = C->trans->transferLayer[i];
			C->trans->transferLayer[i] = Initialize();
			Position Tmp;
			Tmp = Ptemp;
			while (Tmp->Children[0] != NULL)
				Tmp = Tmp->Children[0];
			/* ��һInsert */
			while (Tmp != NULL)
			{
				float cdfcl = cdfCalculate(C->Mnet, Tmp->Key[0]);
				float cdfcu = cdfCalculate(C->Mnet, Tmp->Key[Tmp->KeyNum - 1]);
				int belongcl = cdfcl / (1.0 / C->trans->curnum);
				int belongcu = cdfcu / (1.0 / C->trans->curnum);
				if (belongcl == belongcu)
				{
					for (int ki = 0; ki <= Tmp->KeyNum - 1; ki++)
					{
						C->trans->transferLayer[belongcl] = Insert(C->trans->transferLayer[belongcl], Tmp->Key[ki]);
					}
				}
				else
				{
					for (int ki = 0; ki <= Tmp->KeyNum - 1; ki++)
					{
						cdfcl = cdfCalculate(C->Mnet, Tmp->Key[ki]);
						belongcl = cdfcl / (1.0 / C->trans->curnum);
						C->trans->transferLayer[belongcl] = Insert(C->trans->transferLayer[belongcl], Tmp->Key[ki]);
					}
				}
				Tmp = Tmp->Next;
			}
			Destroy(Ptemp);
			// Travel(C->trans->transferLayer[i]);
			LastNode = NULL;
		}
	}
	cout << "modifiedTrees:" << modifiedTrees << endl;
	LastNode = NULL;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		BPlusTree tmp = C->trans->transferLayer[i];
		Travel(tmp);
		LastNode = NULL;
	}

	CardIndexReport(C);
}
void brief(CardIndex *C)
{
	int tol = 0;
	for (int i = 0; i < C->trans->curnum; i++)
	{
		Position Tmp = C->trans->transferLayer[i];
		while (Tmp->Children[0] != NULL)
		{
			Tmp = Tmp->Children[0];
		}
		while (Tmp != NULL)
		{
			tol += Tmp->KeyNum;
			Tmp = Tmp->Next;
		}
	}

	cout << "\nNum:" << tol << endl;
}
Querys *readQueryFile(string queryfilename)
{
	ifstream infile(queryfilename);
	if (!infile.is_open())
	{
		cout << queryfilename << endl;
		cout << "Fail to load" << endl;
		return NULL;
	}
	int colNumber, queryNumber;
	infile >> colNumber >> queryNumber;
	// cout<<colNumber<< "  "<<queryNumber<<endl;
	Querys *A = new Querys;
	A->queryNumber = queryNumber;
	A->Qs = new Query[queryNumber];
	int *binaryLength = new int[queryNumber];
	for (int i = 0; i < colNumber; i++)
	{
		infile >> binaryLength[i];
	}
	for (int i = 0; i < queryNumber; i++)
	{
		A->Qs[i].binaryLength = binaryLength;
		A->Qs[i].columnNumber = colNumber;
		A->Qs[i].queryid = i;
		A->Qs[i].leftupBound = new long long[colNumber];
		A->Qs[i].rightdownBound = new long long[colNumber];
		for (int j = 0; j < colNumber; j++)
		{
			infile >> A->Qs[i].leftupBound[j];
		}
		for (int j = 0; j < colNumber; j++)
		{
			infile >> A->Qs[i].rightdownBound[j];
		}
		long Tnumber;
		infile >> Tnumber;
		qid2TrueNumber[i] = Tnumber;
		// cout<<Tnumber<<endl;
	}
	// cout << "end" << endl;
	return A;
}
bool *QueryUp2Zvalue(Query Qi, int *tolbitsx, int rightupflag)
{
	// ��Qi�Ķ˵�תZֵ��ͬʱ��λ��0���,rightupflag=1�����ϣ�=0,����
	vector<vector<int>> bCs;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		vector<int> bvi;
		int digitTollen = Qi.binaryLength[i];
		int *vi = new int[digitTollen + 1];

		long long v;
		if (rightupflag == 0)
		{
			v = Qi.leftupBound[i];
		}
		else
		{
			v = Qi.rightdownBound[i];
		}
		longlong2digVec(v, vi, digitTollen);

		for (int ix = 0; ix < digitTollen; ix += 1)
		{
			bvi.push_back(vi[ix]);
		}
		// cout << endl
		//      << v << " " << digitTollen << endl;
		bCs.push_back(bvi);
	}
	int tolbits = 0;
	int maxbit = -1;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		if (Qi.binaryLength[i] > maxbit)
		{
			maxbit = Qi.binaryLength[i];
		}
		tolbits += Qi.binaryLength[i];
	}
	*tolbitsx = tolbits;
	bool *zencode = new bool[tolbits + 1];
	zencode[0] = 0;
	int cnt = 1;
	for (int i = 0; i < maxbit; i++)
	{
		for (int j = 0; j < Qi.columnNumber; j++)
		{
			if (i >= Qi.binaryLength[j])
			{
				continue;
			}
			else
			{
				zencode[cnt] = (bool)bCs[j][i];
				cnt += 1;
			}
		}
	}
	return zencode;
}

// �ݹ������ݣ�����Ҷ�ڵ�ָ��
Position recrusiveFind(BPlusTree T, KeyType Key)
{
	// cout << T << " " << T->KeyNum << endl;
	if (T == NULL)
	{
		// cout << "NUL PTR" << endl;
		return T;
	}

	if (T->Children[0] == NULL)
	{
		return T;
	}
	// cout << "Search begin: " << Key->z32 << " " << T->KeyNum << endl;
	int j, Limit;
	Position Sibling;
	/* ���ҷ�֧ */
	int childIdx = LinearNodeSearch(T, Key);
	// cout << childIdx << endl;
	// if (childIdx == -1)
	// {
	//     childIdx = 0;
	// }
	// if(childIdx >= T->KeyNum){
	//     childIdx = T->KeyNum-1;
	// }
	return recrusiveFind(T->Children[childIdx], Key);

	j = 0;
	while (j < T->KeyNum && (unsigned)Key->z32 >= (unsigned)T->Key[j]->z32)
	{
		if ((unsigned)Key->z32 == (unsigned)T->Key[j]->z32)
		{
			if (T->Children[j] != NULL)
			{
				return recrusiveFind(T->Children[j], Key);
			}
			else
			{
				return T;
			}
		}
		j++;
	}
	// cout<<j<<endl;
	if (j != 0 && T->Children[0] != NULL)
	{
		j--;
	}
	return recrusiveFind(T->Children[j], Key);
}

void testBPlusPointQuery(BPlusTree T, string queryfilepath)
{
	typedef std::chrono::high_resolution_clock Clock;
	int NonleafCnt = 0;
	Querys *qs = readQueryFile(queryfilepath);
	cout << "Doing Point Q" << endl;
	int loopUp = 1;
	for (int loop = 0; loop < loopUp; loop++)
	{
		for (int i = 0; i < qs->queryNumber; i++)
		{
			// i = 67;
			// cout << "\rQid:" << i;
			int scanneditem = 0;
			Query qi = qs->Qs[i];
			bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 1);
			// my turn
			ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
			// cout<<ZT0->z32<<endl;exit(1);
			auto t1 = Clock::now(); // ��ʱ��ʼ
			Position p0 = recrusiveFind(T, ZT0);
			int flag = 0;
			for (int j = 0; j < p0->KeyNum; j++)
			{
				// cout<<p0->Key[j]->z32<<" "<<ZT0->z32<<endl;
				if (p0->Key[j]->z32 == ZT0->z32)
				{
					cout << "Find" << endl;
					flag = 1;
					break;
				}

				// cout<<p0->Key[j]->z32<<" "<<ZT0->z32<<endl;
			}
			if (flag == 0)
			{
				cout << "nf" << endl;
				exit(1);
			}
			// exit(1);
			// int *blk0s = pointQueryTriple(M, qi, zencode0);
			auto t1d = Clock::now(); // ��ʱ��ʼ
			NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1d - t1).count());
			continue;
		}
	}
	cout << "Avg:" << NonleafCnt / (0.0 + loopUp * qs->queryNumber) << endl;
}

bool inboxZ(Query Qi, bool *zencode)
{
	int maxlen = 0;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		long long chkv = 0;
		for (auto idx : Zidx2Col[i])
		{
			chkv *= 2;
			chkv += zencode[idx];
		}
		if (chkv < Qi.leftupBound[i] || chkv > Qi.rightdownBound[i])
		{
			return false;
		}
	}
	return true;
	long long zdecode[20] = {0};
	for (int i = 0; i < Qi.columnNumber; i += 1)
	{
		zdecode[i] = 0;
		if (Qi.binaryLength[i] > maxlen)
		{
			maxlen = Qi.binaryLength[i];
		}
	}

	int idx = 0;
	// cout << maxlen << endl;
	for (int i = 0; i < maxlen; i += 1)
	{
		for (int j = 0; j < Qi.columnNumber; j++)
		{
			if (i >= Qi.binaryLength[j])
			{
				continue;
			}
			else
			{
				zdecode[j] *= 2;
				zdecode[j] += zencode[idx];
				idx += 1;
			}
		}
	}
	// cout << idx << endl;

	/*cout << "BL" << endl;
	for (int j = 0; j < Qi.columnNumber; j++) {
		cout << Qi.binaryLength[j] << " ";
	}cout << endl;
	cout << "LU:";

	for (int j = 0; j < Qi.columnNumber; j++) {
		cout << Qi.leftupBound[j]<< " ";
	}cout << endl;
	cout << "ck:";
	for (int j = 0; j < Qi.columnNumber; j++) {
		cout << zdecode[j] << " ";
	}cout << endl;
	cout << "RD:";
	for (int j = 0; j < Qi.columnNumber; j++) {
		cout << Qi.rightdownBound[j] << " ";
	}cout << endl;*/

	for (int i = 0; i < Qi.columnNumber; i++)
	{
		if (zdecode[i] < Qi.leftupBound[i] || zdecode[i] > Qi.rightdownBound[i])
		{
			return false;
		}
	}
	return true;
}
bool inbox(Query *qi, KeyType Key)
{
	int c = qi->columnNumber;
	int preserve = 0;
	for (int i = 0; i < c; i++)
	{
		long long v = Key->values[i];
		// cout<<qi->leftupBound[i]<<" "<<v<<" "<<qi->rightdownBound[i]<<endl;
		if (v > qi->rightdownBound[i])
		{
			return false;
		}
		else if (v < qi->leftupBound[i])
		{
			return false;
		}
	}
	return true;
}
int maxlen = 0;

bool *getLITMAX(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
	bool *Litmax = new bool[bitlength];
	bool *tmpmaxZ = new bool[bitlength];
	bool *tmpminZ = new bool[bitlength];
	for (int i = 0; i < bitlength; i += 1)
	{
		tmpminZ[i] = minZ[i];
		tmpmaxZ[i] = maxZ[i];
		Litmax[i] = 0;
	}
	/*int maxlen = 0;
	for (int i = 0; i < colNum; i += 1)
	{
		if (colbitlength[i] > maxlen)
		{
			maxlen = colbitlength[i];
		}
	}*/
	int idx = 0;
	for (int i = 0; i < maxlen; i += 1)
	{
		for (int j = 0; j < colNum; j++)
		{
			if (i >= colbitlength[j])
			{
				continue;
			}
			else
			{
				// cout<<i<<"th dig of col"<<j<<endl;
				bool divnum = zvalue[idx];
				bool minnum = tmpminZ[idx];
				bool maxnum = tmpmaxZ[idx];
				// cout << "Idx" << idx << " div:" << divnum << " minnum:" << minnum << "  maxN:" << maxnum << endl;
				// cout << "Tminz:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << tmpminZ[i] << " ";
				// }
				// cout << endl;
				// cout << "Tmaxz:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << tmpmaxZ[i] << " ";
				// }
				// cout << endl;
				// cout << "Bigmi:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << Bigmin[i] << " ";
				// }
				// cout << endl;

				if (divnum == 0 && minnum == 0 && maxnum == 0)
				{
					idx += 1;
					continue;
				}
				if (divnum == 1 && minnum == 1 && maxnum == 1)
				{
					idx += 1;
					continue;
				}
				if (divnum == 0 && minnum == 1 && maxnum == 1)
				{
					// cout<<"code:011"<<endl;
					return Litmax;
				}
				if (divnum == 1 && minnum == 0 && maxnum == 0)
				{
					// cout<<"code100"<<endl;
					return tmpmaxZ;
				}
				if (divnum == 0 && minnum == 1 && maxnum == 0)
				{
					return zvalue;
					cout << "LITMAXWRONG!" << endl;
					exit(1);
					// return Bigmin;
				}
				if (divnum == 1 && minnum == 1 && maxnum == 0)
				{
					return zvalue;
					cout << "LITMAXWRONG!" << endl;
					exit(1);
					// return Bigmin;
				}

				if (divnum == 0 && minnum == 0 && maxnum == 1)
				{
					// cout << "CODE: 101" << endl;
					// max = 1000000
					int innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										tmpmaxZ[innercnt] = 0;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										tmpmaxZ[innercnt] = 1;
									}
									innercnt += 1;
								}
							}
						}
					}
					// Bigmin = tmpminZ;
					idx += 1;
					continue;
				}
				if (divnum == 1 && minnum == 0 && maxnum == 1)
				{
					// cout << "CODE:001" << endl;
					for (int x00 = 0; x00 < bitlength; x00++)
					{
						Litmax[x00] = tmpmaxZ[x00];
					}
					int innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										Litmax[innercnt] = 0;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										Litmax[innercnt] = 1;
									}
									innercnt += 1;
								}
							}
						}
					}
					innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										tmpminZ[innercnt] = 1;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										tmpminZ[innercnt] = 0;
									}
									innercnt += 1;
								}
							}
						}
					}
					// cout << "Tminz:";
					// for (int i = 0; i < bitlength; i++)
					// {
					//     cout << tmpminZ[i];
					// }
					// cout << endl;
					// cout << "Tmaxz:";
					// for (int i = 0; i < bitlength; i++)
					// {
					//     cout << tmpmaxZ[i];
					// }
					// cout << endl;
					// cout << "Bigmi:";
					// for (int i = 0; i < bitlength; i++)
					// {
					//     cout << Bigmin[i];
					// }
					// cout << endl;
					idx += 1;
					continue;
				}
				idx += 1;
			}
		}
	}
	// cout<<"Normal Ret"<<endl;
	return Litmax;
}

bool *getBIGMIN(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
	// ���룺��ѯ�������minz������maxz����Χ���zvalue������zvalue��bigmin
	bool *Bigmin = new bool[bitlength];
	bool *tmpmaxZ = new bool[bitlength];
	bool *tmpminZ = new bool[bitlength];
	// bool Bigmin[120] = { 0 };
	// bool tmpmaxZ [120];
	// bool tmpminZ [120];
	memcpy(tmpmaxZ, maxZ, bitlength);
	memcpy(tmpminZ, minZ, bitlength);
	memset(Bigmin, 0, bitlength);
	// for (int i = 0; i < bitlength; i += 1)
	//{
	//	tmpminZ[i] = minZ[i];
	//	tmpmaxZ[i] = maxZ[i];
	//	Bigmin[i] = 0;
	// }
	//  cout << "iptzv:";
	//  for (int i = 0; i < bitlength - 1; i++)
	//  {
	//      cout << zvalue[i];
	//  }
	//  cout << endl;
	//  cout << "Tminz:";
	//  for (int i = 0; i < bitlength - 1; i++)
	//  {
	//      cout << tmpminZ[i];
	//  }
	//  cout << endl;
	//  cout << "Tmaxz:";
	//  for (int i = 0; i < bitlength - 1; i++)
	//  {
	//      cout << tmpmaxZ[i];
	//  }
	//  cout << endl;

	int idx = 0;
	for (int i = 0; i < maxlen; i += 1)
	{
		for (int j = 0; j < colNum; j++)
		{
			if (i >= colbitlength[j])
			{
				continue;
			}
			else
			{
				// cout<<i<<"th dig of col"<<j<<endl;
				bool divnum = zvalue[idx];
				bool minnum = tmpminZ[idx];
				bool maxnum = tmpmaxZ[idx];
				// cout << "Idx" << idx << " div:" << divnum << " minnum:" << minnum << "  maxN:" << maxnum << endl;
				// cout << "Tminz:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << tmpminZ[i] << " ";
				// }
				// cout << endl;
				// cout << "Tmaxz:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << tmpmaxZ[i] << " ";
				// }
				// cout << endl;
				// cout << "Bigmi:";
				// for (int i = 0; i < bitlength; i++)
				// {
				//     cout << Bigmin[i] << " ";
				// }
				// cout << endl;

				if (divnum == 0 && minnum == 0 && maxnum == 0)
				{
					idx += 1;
					continue;
				}
				if (divnum == 1 && minnum == 1 && maxnum == 1)
				{
					idx += 1;
					continue;
				}
				if (divnum == 0 && minnum == 1 && maxnum == 1)
				{
					// cout<<"code:011"<<endl;
					return tmpminZ;
				}
				if (divnum == 1 && minnum == 0 && maxnum == 0)
				{
					// cout<<"code100"<<endl;
					return Bigmin;
				}
				if (divnum == 0 && minnum == 1 && maxnum == 0)
				{
					return zvalue;
					// cout << "BMWRONG!" << endl;
					// exit(1);
					// return Bigmin;
				}
				if (divnum == 1 && minnum == 1 && maxnum == 0)
				{
					return zvalue;
					// cout << "BMWRONG!" << endl;
					// exit(1);
					// return Bigmin;
				}

				if (divnum == 1 && minnum == 0 && maxnum == 1)
				{
					// cout << "CODE: 101" << endl;
					// minz = 1000000
					int innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										tmpminZ[innercnt] = 1;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										tmpminZ[innercnt] = 0;
									}
									innercnt += 1;
								}
							}
						}
					}
					// Bigmin = tmpminZ;
					idx += 1;
					continue;
				}
				if (divnum == 0 && minnum == 0 && maxnum == 1)
				{
					// cout << "CODE:001" << endl;
					for (int x00 = 0; x00 < bitlength; x00++)
					{
						Bigmin[x00] = tmpminZ[x00];
					}
					int innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										Bigmin[innercnt] = 1;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										Bigmin[innercnt] = 0;
									}
									innercnt += 1;
								}
							}
						}
					}
					innercnt = 0;
					for (int x0 = 0; x0 < maxlen; x0++)
					{
						for (int x1 = 0; x1 < colNum; x1 += 1)
						{
							if (x0 >= colbitlength[x1])
							{
								continue;
							}
							else
							{
								if (x0 < i)
								{
									innercnt++;
									continue;
								}
								else if (x0 == i)
								{
									if (x1 == j)
									{
										tmpmaxZ[innercnt] = 0;
									}
									innercnt += 1;
								}
								else
								{
									if (x1 == j)
									{
										tmpmaxZ[innercnt] = 1;
									}
									innercnt += 1;
								}
							}
						}
					}
					idx += 1;
					continue;
				}
				idx += 1;
			}
		}
	}
	return Bigmin;
}

void buildColPattern(int bitlength, int *colbitlength, int colNum)
{
	int binidx = 0;
	if (Zidx2Col.size() != 0)
	{ // only called once;
		return;
	}
	for (int j = 0; j < colNum; j++)
	{
		Zidx2Col.push_back({});
	}
	for (int i = 0; i < maxlen; i += 1)
	{
		for (int j = 0; j < colNum; j++)
		{
			if (i >= colbitlength[j])
			{
				continue;
			}
			colPattern[binidx] = j;
			Zidx2Col[j].push_back(binidx);
			binidx++;
		}
	}
}
bool *getBIGMINF(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
	// ���룺��ѯ�������minz������maxz����Χ���zvalue������zvalue��bigmin
	bool *Bigmin = new bool[bitlength];
	bool *tmpmaxZ = new bool[bitlength];
	bool *tmpminZ = new bool[bitlength];
	memcpy(tmpmaxZ, maxZ, bitlength);
	memcpy(tmpminZ, minZ, bitlength);
	memset(Bigmin, 0, bitlength);
	int idx = 0;
	for (idx = 0; idx < *tolbits; idx++)
	{
		bool divnum = zvalue[idx];
		bool minnum = tmpminZ[idx];
		bool maxnum = tmpmaxZ[idx];
		if (divnum == 0 && minnum == 0 && maxnum == 0)
		{
			continue;
		}
		if (divnum == 1 && minnum == 1 && maxnum == 1)
		{
			continue;
		}
		if (divnum == 0 && minnum == 1 && maxnum == 1)
		{
			return tmpminZ;
		}
		if (divnum == 1 && minnum == 0 && maxnum == 0)
		{
			return Bigmin;
		}
		if (divnum == 0 && minnum == 1 && maxnum == 0)
		{
			cout << "BMWRONG!" << endl;
			exit(1);
			return zvalue;
		}
		if (divnum == 1 && minnum == 1 && maxnum == 0)
		{
			cout << "BMWRONG!" << endl;
			exit(1);
			return zvalue;
		}
		if (divnum == 1 && minnum == 0 && maxnum == 1)
		{ // minz = 1000000
			int belongNum = colPattern[idx];
			vector<int> bitlocation = Zidx2Col[belongNum];
			int setbit = 1;
			for (auto zidxi : bitlocation)
			{
				if (zidxi < idx)
				{
					continue;
				}
				else if (zidxi == idx)
				{
					tmpminZ[zidxi] = setbit;
					setbit = 0;
				}
				else
				{
					tmpminZ[zidxi] = setbit;
				}
			}
			continue;
		}
		if (divnum == 0 && minnum == 0 && maxnum == 1)
		{
			// cout << "CODE:001" << endl;
			memcpy(Bigmin, tmpminZ, bitlength);
			// load 10000 bigmin
			// load 01111 tmpmax
			int belongNum = colPattern[idx];
			vector<int> bitlocation = Zidx2Col[belongNum];
			bool setbitBigmin = 1;
			bool settmpmax = 0;
			for (auto zidxi : bitlocation)
			{
				if (zidxi < idx)
				{
					continue;
				}
				else if (zidxi == idx)
				{
					Bigmin[zidxi] = setbitBigmin;
					tmpmaxZ[zidxi] = settmpmax;
					setbitBigmin = 0;
					settmpmax = 1;
				}
				else
				{
					Bigmin[zidxi] = setbitBigmin;
					tmpmaxZ[zidxi] = settmpmax;
				}
			}
		}
	}

	return Bigmin;
}
bool *getLITMAXF(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
	// ���룺��ѯ�������minz������maxz����Χ���zvalue������zvalue��bigmin
	bool *Litmax = new bool[bitlength];
	bool *tmpmaxZ = new bool[bitlength];
	bool *tmpminZ = new bool[bitlength];
	memcpy(tmpmaxZ, maxZ, bitlength);
	memcpy(tmpminZ, minZ, bitlength);
	memset(Litmax, 0, bitlength);
	int idx = 0;
	for (idx = 0; idx < *tolbits; idx++)
	{
		bool divnum = zvalue[idx];
		bool minnum = tmpminZ[idx];
		bool maxnum = tmpmaxZ[idx];
		if (divnum == 0 && minnum == 0 && maxnum == 0)
		{
			continue;
		}
		if (divnum == 1 && minnum == 1 && maxnum == 1)
		{
			continue;
		}
		if (divnum == 0 && minnum == 1 && maxnum == 1)
		{
			return Litmax;
		}
		if (divnum == 1 && minnum == 0 && maxnum == 0)
		{
			return tmpmaxZ;
		}
		if (divnum == 0 && minnum == 1 && maxnum == 0)
		{
			cout << "LMWRONG!" << endl;
			exit(1);
			return zvalue;
		}
		if (divnum == 1 && minnum == 1 && maxnum == 0)
		{
			cout << "LMWRONG!" << endl;
			exit(1);
			return zvalue;
		}
		if (divnum == 0 && minnum == 0 && maxnum == 1)
		{ // minz = 1000000
			int belongNum = colPattern[idx];
			vector<int> bitlocation = Zidx2Col[belongNum];
			int setbit = 0;
			for (auto zidxi : bitlocation)
			{
				if (zidxi < idx)
				{
					continue;
				}
				else if (zidxi == idx)
				{
					tmpmaxZ[zidxi] = setbit;
					setbit = 1;
				}
				else
				{
					tmpmaxZ[zidxi] = setbit;
				}
			}
			continue;
		}
		if (divnum == 1 && minnum == 0 && maxnum == 1)
		{
			// cout << "CODE:101" << endl;
			memcpy(Litmax, tmpmaxZ, bitlength);
			// load 10000 bigmin
			// load 01111 tmpmax
			int belongNum = colPattern[idx];
			vector<int> bitlocation = Zidx2Col[belongNum];
			bool setbitlitmax = 0;
			bool settmpmin = 1;
			for (auto zidxi : bitlocation)
			{
				if (zidxi < idx)
				{
					continue;
				}
				else if (zidxi == idx)
				{
					Litmax[zidxi] = setbitlitmax;
					tmpminZ[zidxi] = settmpmin;
					setbitlitmax = 1;
					settmpmin = 0;
				}
				else
				{
					Litmax[zidxi] = setbitlitmax;
					tmpminZ[zidxi] = settmpmin;
				}
			}
		}
	}

	return Litmax;
}
int rangeQueryExceute(BPlusTree T, Query qi)
{
	if (T == NULL)
	{
		return 0;
	}
	int scanneditem = 0;
	bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
	bool *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
	ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
	ZTuple *ZT1 = makeZT(&zencode1[1], *tolbits);
	return rangeQueryExceuteF(T, qi, zencode0, zencode1, ZT0, ZT1);
}

int rangeQueryExceuteF(BPlusTree T, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1)
{
	// cout << "p0:" << ZT0->z32 << endl;
	Position p0 = recrusiveFind(T, ZT0);
	// cout << "p1:" << ZT1->z32 << endl;
	Position p1 = recrusiveFind(T, ZT1);
	// cout << "dn" << endl;
	if (p0 == NULL && p1 == NULL)
	{
		return 0;
	}
	int estcard = 0;
	// cout << "counting" << endl;
	for (Position tmp = p0; (tmp != p1->Next) && (tmp != NULL); tmp = tmp)
	{
		lastN = tmp;
		int flag = 0;
		for (int ti = 0; ti < tmp->KeyNum; ti++)
		{
			if (inbox(&qi, tmp->Key[ti]))
			{
				estcard += 1;
				flag = 1;
			}
		}
		if (flag == 0)
		{

			bool *bigmin = getBIGMIN(ZT0->bin, ZT1->bin, tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
			ZTuple *ZTX = makeZT(bigmin, *tolbits);
			Position pt = recrusiveFind(T, ZTX);
			if (pt == NULL)
			{
				tmp = tmp->Next;
			}
			// cout<<pt->Key[0]->z32<<" "<<tmp->Key[0]->z32<<endl;
			if (pt->Key[0]->z32 <= tmp->Key[0]->z32 || pt->Key[0]->z32 > p1->Key[0]->z32)
			{
				tmp = tmp->Next;
			}
			else
			{
				tmp = pt;
			}
		}
		else
		{
			tmp = tmp->Next;
		}
		// if (tmp->Next == NULL || tmp->Next->Key[0]->z32 > p1->Key[0]->z32)
		// {
		//     break;
		// }
	}
	// cout << "over" << endl;
	return estcard;
}

void testBPlusRangeQuery(BPlusTree T, string queryfilepath)
{
	typedef std::chrono::high_resolution_clock Clock;
	long long NonleafCnt = 0;
	Querys *qs = readQueryFile(queryfilepath);
	cout << "Doing Range Q" << endl;
	int loopUp = 1;

	for (int loop = 0; loop < loopUp; loop++)
	{
		for (int i = 0; i < qs->queryNumber; i++)
		{
			// i = 1;
			cout << "Qid:" << i << endl;
			int scanneditem = 0;
			Query qi = qs->Qs[i];
			bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
			bool *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
			// my turn
			ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
			ZTuple *ZT1 = makeZT(&zencode1[1], *tolbits);
			// cout << ZT0->z32 << endl;
			// cout << ZT1->z32 << endl;
			auto t1 = Clock::now(); // ��ʱ��ʼ
			Position p0 = recrusiveFind(T, ZT0);
			// cout << ZT0->z32 << ": ";
			// for (int j = 0; j < qi.columnNumber; j++)
			// {
			//     cout << qi.leftupBound[j] << " ";
			// }
			// cout << endl;
			Position p1 = recrusiveFind(T, ZT1);
			int estcard = 0;
			int truecard = qid2TrueNumber[i];
			// cout << ZT1->z32 << ": ";
			// for (int j = 0; j < qi.columnNumber; j++)
			// {
			//     cout << qi.rightdownBound[j] << " ";
			// }
			// cout << endl;
			// cout << "D------------------" << endl;
			int tolnum = 0;
			for (Position tmp = p0; (tmp != p1->Next) && (tmp != NULL); tmp = tmp)
			{
				// cout<<tmp->Key[0]->z32<<endl;
				int flag = 0;
				for (int ti = 0; ti < tmp->KeyNum; ti++)
				{
					tolnum++;
					// cout<<tmp->Key[ti]->z32<< ": ";
					// for (int j = 0; j < qi.columnNumber; j++)
					// {
					//     cout << tmp->Key[ti]->values[j] << " ";
					// }
					// cout << endl;
					if (inbox(&qi, tmp->Key[ti]))
					{
						estcard += 1;
						flag = 1;
					}
					// exit(1);
				}
				if (flag == 0)
				{
					bool *bigmin = getBIGMIN(ZT0->bin, ZT1->bin, tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
					ZTuple *ZTX = makeZT(bigmin, *tolbits);
					Position pt = recrusiveFind(T, ZTX);
					// cout<<pt->Key[0]->z32<<" "<<tmp->Key[0]->z32<<endl;
					if (pt->Key[0]->z32 <= tmp->Key[0]->z32)
					{
						tmp = tmp->Next;
					}
					else
					{
						tmp = pt;
					}
				}
				else
				{
					tmp = tmp->Next;
				}
				// if (tmp->Next == NULL || tmp->Next->Key[0]->z32 > p1->Key[0]->z32)
				// {
				//     break;
				// }
			}
			cout << estcard << " " << truecard << " scanned:" << tolnum << endl;
			// exit(1);
			// int *blk0s = pointQueryTriple(M, qi, zencode0);
			auto t1d = Clock::now(); // ��ʱ��ʼ
			NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1d - t1).count());
			continue;
		}
		// exit(1);
	}
	cout << "Avg:" << NonleafCnt / (0.0 + loopUp * qs->queryNumber) << endl;
}

float pErrorcalculate(int est, int gt)
{
	if (est < gt)
	{
		int t = gt;
		gt = est;
		est = t;
	}
	if (gt == 0)
	{
		gt = 1;
	}
	return (est + 0.0) / (gt + 0.0);
}
void MadeIndexInferDig(int *xinput, float *out, int startidx, int endidx, MADENet *net, float *middle)
{
	int winlen = net->connectlen;
	for (int i = startidx; i <= endidx; i++)
	{
		middle[i] = net->fc1b[i];
		for (int j = max(i - winlen, 0); j < i; j++)
		{
			if (j >= i)
			{
				break;
			}
			middle[i] += (xinput[j] * net->fc1w[i * net->diglen + j]);
		}
		if (middle[i] < 0)
		{
			middle[i] = 0;
		}
	}
	// for (int i=0;i<5;i++){
	//     cout<<middle[i]<<" ";
	// }cout<<endl;
	for (int i = startidx; i <= endidx; i++)
	{
		out[i] = net->fc2b[i];
		// cout<<"oi"<<out[i]<<endl;
		for (int j = max(i - winlen, 0); j < i; j++)
		{
			if (j >= i)
			{
				break;
			}
			out[i] += (middle[j] * net->fc2w[i * net->diglen + j]);
		}
		// cout<<out[i]<<endl;
		out[i] = (1.0) / (1.0 + exp(-out[i]));
		// cout<<out[i]<<endl;
	}
	// for(int i=0;i<5;i++){
	//     cout<<out[i]<<endl;
	// }
	// exit(1);
}

int lrpermitCheck(long long minv, long long maxv, int encodeLength, int position, long long *rec, int demo)
{
	// cout<<"I'm in"<<endl;
	if (demo == 1)
	{
		cout << "MIN: " << minv << " MAX: " << maxv << " LEN:" << encodeLength << endl;
	}
	//
	bool tmpminv[200];
	bool tmpmaxv[200];
	long long tminv = minv;
	long long tmaxv = maxv;
	for (int i = 0; i < encodeLength; i++)
	{
		tmpminv[encodeLength - i - 1] = tminv % 2;
		tmpmaxv[encodeLength - i - 1] = tmaxv % 2;
		// cout<<tmaxv<<" "<< encodeLength-i<<" "<<tmpmaxv[encodeLength-i-1] <<endl;
		tminv = tminv >> 1;
		tmaxv = tmaxv >> 1;
	}
	if (tmaxv != 0)
	{
		for (int i = 0; i < encodeLength; i++)
		{
			tmpmaxv[encodeLength - i - 1] = 1;
		}
	}
	// cout<<tmaxv<<endl;
	// cout<<"\n";
	if (demo == 1)
	{
		for (int i = 0; i < encodeLength; i++)
		{
			cout << tmpmaxv[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < encodeLength; i++)
		{
			cout << tmpminv[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < position; i++)
		{
			cout << rec[i] << " ";
		}
		cout << endl;
	}
	//
	int leftp = 0; // 0:Nan,1:leftPermit
	int flagx0 = 1;
	for (int i = 0; i < position; i++)
	{
		if (rec[i] > tmpminv[i])
		{
			leftp = 1;
			flagx0 = 0;
			break;
		}
		else if (rec[i] < tmpminv[i])
		{
			leftp = 0;
			flagx0 = 0;
			break;
		}
	}
	if (flagx0 == 1)
	{
		if (tmpminv[position] == 0)
		{
			leftp = 1;
		}
	}
	flagx0 = 1;		// reset
	int rightp = 0; // 0:Nan,1:rightPermit
	for (int i = 0; i < position; i++)
	{
		if (rec[i] > tmpmaxv[i])
		{
			rightp = 0;
			flagx0 = 0;
			break;
		}
		else if (rec[i] < tmpmaxv[i])
		{
			rightp = 1;
			flagx0 = 0;
			break;
		}
	}
	if (flagx0 == 1)
	{
		if (tmpmaxv[position] == 1)
		{
			rightp = 1;
		}
	}
	int retv = 0; // 0:AllNon , 1: Left P,2: rightP,3:AllP
	if (leftp == 0 && rightp == 0)
	{
		retv = 0;
	}
	if (leftp == 1 && rightp == 0)
	{
		retv = 1;
	}
	if (leftp == 0 && rightp == 1)
	{
		retv = 2;
	}
	if (leftp == 1 && rightp == 1)
	{
		retv = 3;
	}

	// cout<<retv<<endl;
	return retv;
}
float drawZF(MADENet *root, Query Qi, int demo)
{
	float out[150] = {0};
	float mid[150] = {0};
	float p = 1.0;
	int *binaryList = Qi.binaryLength;
	int binaryAllLen = 0;
	int maxBinecn = -1;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		binaryAllLen += binaryList[i];
	}
	int currentdepth = 0;
	long long searchState[15] = {0};
	int stateBit[15] = {0};
	int flag = 0;
	int samplePoint[150] = {0};
	int layer = 0;
	for (int ix = 0; ix < min(binaryAllLen, 50); ix++)
	{
		int colBelong = colPattern[ix];
		int colDig = stateBit[colBelong];
		int colMaxv = Qi.rightdownBound[colBelong];
		int colMinv = Qi.leftupBound[colBelong];
		int colTolbits = Qi.binaryLength[colBelong];
		int shiftN = (colTolbits - colDig - 1);
		int tmpMaxv = colMaxv >> shiftN;
		int tmpMinv = colMinv >> shiftN;
		int curval = searchState[colBelong] >> (shiftN + 1);
		int leftp = 0;
		int rightp = 0;
		float OneProb;
		MadeIndexInferDig(samplePoint, out, ix, ix, root, mid);
		OneProb = out[ix];
		// cout << "Bit:" << ix << " Col" << colBelong << " Min:" << colMinv << " Max: " << colMaxv << " tmpmin:" << tmpMinv << " tmpmax: " << tmpMaxv << endl;
		// cout << "Colinfo: " << colTolbits << " " << colDig << endl;
		// cout << searchState[colBelong] << " " << curval << endl;
		if (curval * 2 + 1 <= tmpMaxv)
		{
			rightp = 1;
		}
		if (curval * 2 >= tmpMinv)
		{
			leftp = 1;
		}
		// cout << "L: " << leftp << "R: " << rightp << endl;
		if (leftp == 0 && rightp == 0)
		{
			// not possible
			p = 0;
			cout << "p0" << endl;
			exit(1);
		}
		if (leftp == 1 && rightp == 0)
		{
			// zero
			samplePoint[ix] = 0;
			searchState[colBelong] = (curval << 1) << (shiftN);
			p = p * (1 - OneProb);
		}
		if (leftp == 0 && rightp == 1)
		{
			// one
			samplePoint[ix] = 1;
			searchState[colBelong] = ((curval << 1) + 1) << (shiftN);
			p = p * OneProb;
		}
		if (leftp == 1 && rightp == 1)
		{
			samplePoint[ix] = randG(OneProb);
			searchState[colBelong] = ((curval << 1) + samplePoint[ix]) << (shiftN);
		}
		stateBit[colBelong]++;
	}
	// exit(1);
	return p;
}

float drawZ(MADENet *root, Query Qi, int demo)
{
	float out[150];
	float mid[150];
	float p = 1.0;
	int *binaryList = Qi.binaryLength;
	int binaryAllLen = 0;
	int maxBinecn = -1;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		if (binaryList[i] > maxBinecn)
		{
			maxBinecn = binaryList[i];
		}
		if (binaryList[i] == 0)
		{
			binaryAllLen += 1;
			continue;
		}
		binaryAllLen += binaryList[i];
	}
	int currentdepth = 0;
	long long *searchState[15];
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		int binlen = binaryList[i] + 1;
		searchState[i] = new long long[binlen];
	}
	int flag = 0;
	int samplePoint[150];
	for (int i = 0; i < binaryAllLen; i++)
	{
		samplePoint[i] = 0;
	}
	float OneProbpath[150];
	int innerloopCounter = 0;
	int layer = 0;
	// cout<<maxBinecn<<Qi.columnNumber<<endl;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		// cout<< binaryList[i]<<" ";
		if (binaryList[i] == 0)
		{
			binaryList[i] = 1;
		}
	}

	for (int i = 0; i < 100; i++)
	{
		samplePoint[i] = 0;
		out[i] = 0;
		mid[i] = 0;
	}
	for (int ix = 0; ix < maxBinecn; ix++)
	{
		for (int j = 0; j < Qi.columnNumber; j++)
		{

			if (ix >= binaryList[j])
			{
				continue;
			}
			else
			{
				// cout<<currentdepth<<endl;
				// currentdepth+=1;
				// continue;
				// eval

				// cout << samplePoint << endl;
				if (demo == 1)
				{
					cout << ix << " th Dig of col" << j << endl;
					cout << "Layer:" << layer << endl;
				}
				// cout<<ix<<" th Dig of col"<<j<<endl;
				float OneProb;

				// cout << currentdepth<<endl;
				//  cout<<"curDep"<<currentdepth<<" defaultDepth:"<<defaultDepth<<endl;
				currentdepth += 1;
				if (currentdepth >= 50)
				{
					return p;
				}

				MadeIndexInferDig(samplePoint, out, innerloopCounter, innerloopCounter, root, mid);
				OneProb = out[innerloopCounter];
				// cout<<innerloopCounter <<" "<<OneProb<<endl;
				// exit(1);
				OneProbpath[innerloopCounter] = OneProb;
				// if (innerloopCounter > 80)
				// {
				//     return p;
				// }
				if (demo == 1)
				{
					cout << "ilc" << innerloopCounter << " NetInput:" << samplePoint[innerloopCounter] << endl;
					cout << "OneProb: " << OneProb << " P: " << p << endl;
					// if (innerloopCounter ==3){
					//     exit(1);
					// }
				}
				// cout<<"OneProb: "<<OneProb<<" P: "<<p<<endl;
				// ���ݲ�ѯ��֦
				int ff = lrpermitCheck(Qi.leftupBound[j], Qi.rightdownBound[j], Qi.binaryLength[j], ix, searchState[j], demo);
				if (demo == 1)
				{
					cout << "PermitState:" << ff << endl;
				}
				//

				if (ff == 0)
				{
					p = 0;
					cout << "p0" << endl;
					exit(1);
					// return 0;
				}
				if (ff == 1)
				{
					samplePoint[innerloopCounter] = 0;
					searchState[j][ix] = 0;
					p = p * (1 - OneProb);
				}
				if (ff == 2)
				{
					samplePoint[innerloopCounter] = 1;
					searchState[j][ix] = 1;
					p = p * OneProb;
				}
				if (ff == 3)
				{ // ���ɲ�����
					samplePoint[innerloopCounter] = randG(OneProb);
					searchState[j][ix] = samplePoint[innerloopCounter];
				}
				innerloopCounter += 1;
			}
		}
	}
	// cout<<"est done"<<endl;
	// for (int i = 0; i < 20; i++)
	// {
	//     cout << samplePoint[i] << " ";
	// }
	// for (int i = 0; i < 20; i++)
	// {
	//     cout << out[i] << " ";
	// }

	// cout<<"Over"<<endl;
	// cout<<p<<endl;
	// exit(1) ;
	for (int i = 0; i < Qi.columnNumber; i++)
	{
		delete searchState[i];
	}
	return p;

	return 0.0;
}

void threadRun(int idx, int sampleNumber)
{
	MADENet *root = MadeBuffer[idx];
	// cout<<root->connectlen<<" "<<root->diglen<<" "<<root->zdr<<endl;
	Query *Qi = QueryBuffer[idx];
	// cout<<Qi->queryid<<" "<<Qi->columnNumber<<endl;
	float p = 0;
	int demo = 0;
	for (int i = 0; i < sampleNumber; i++)
	{
		p += drawZF(root, *Qi, demo);
		// cout<<p<<endl;
		if (demo == 1)
		{
			cout << p << endl;
			exit(1);
		}
		/* code */
	}
	p = p / (sampleNumber + 0.0);
	// cout<<p<<endl;
	// cout<<p* root->zdr<<endl;
	tret[idx] = p;
}

int cardEstimate(MADENet *root, Query Qi, int sampleNumber)
{
	// tret[0]=0;
	int paraNum = 16;
	for (int i = 1; i <= paraNum; i++)
	{
		tret[i] = 0;
	}

	float p = 0;
	int demo = 0;

	thread t1(threadRun, 1, sampleNumber / paraNum);
	thread t2(threadRun, 2, sampleNumber / paraNum);
	thread t3(threadRun, 3, sampleNumber / paraNum);
	thread t4(threadRun, 4, sampleNumber / paraNum);
	thread t5(threadRun, 5, sampleNumber / paraNum);
	thread t6(threadRun, 6, sampleNumber / paraNum);
	thread t7(threadRun, 7, sampleNumber / paraNum);
	thread t8(threadRun, 8, sampleNumber / paraNum);
	thread t9(threadRun, 9, sampleNumber / paraNum);
	thread t10(threadRun, 10, sampleNumber / paraNum);
	thread t11(threadRun, 11, sampleNumber / paraNum);
	thread t12(threadRun, 12, sampleNumber / paraNum);
	thread t13(threadRun, 13, sampleNumber / paraNum);
	thread t14(threadRun, 14, sampleNumber / paraNum);
	thread t15(threadRun, 15, sampleNumber / paraNum);
	thread t16(threadRun, 16, sampleNumber / paraNum);

	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();
	t8.join();
	t9.join();
	t10.join();
	t11.join();
	t12.join();
	t13.join();
	t14.join();
	t15.join();
	t16.join();
	// for (int i = 0; i < sampleNumber; i++)
	// {
	//     p += drawZ(root, Qi, demo);
	//     if (demo == 1)
	//     {
	//         cout << p << endl;
	//         exit(1);
	//     }
	//     /* code */
	// }
	// p/=sampleNumber;
	// p =(tret[1]);
	for (int i = 1; i <= paraNum; i++)
	{
		p += tret[i];
	}
	p /= paraNum;
	// cout<<"Outside:"<<p<<endl;
	return int(root->zdr * (p));
}
float point2cdfest(MADENet *root, Query Qi, bool *zencode)
{
	float cdf = 0;
	float acc = 1.0;
	float OneProb;
	int belong;
	float minnimumsep;
	float out[35];
	MadeIndexInfer(zencode, out, infL, root, midlle);
	for (int i = 0; i < infL; i++)
	{

		OneProb = out[i];
		cdf += (acc * (1 - OneProb) * zencode[i]);
		acc *= (OneProb * zencode[i] + (1 - OneProb) * (1 - zencode[i]));
		// cout<<i<<OneProb<<" "<<cdf<<" "<<acc<<endl;
	}
	return cdf;
}

int getBelongNum(CardIndex *C, ZTuple *zti)
{
	float cdf = cdfCalculate(C->Mnet, zti);
	if (cdf >= 1)
	{
		cdf = 1;
	}
	int belong = cdf / (1.0 / C->trans->curnum);
	if (belong <= 0)
	{
		belong = 0;
	}
	if (belong >= MAXL1CHILD)
	{
		belong = MAXL1CHILD - 1;
	}
	return belong;
}
void ZADD(bool *zencode0, bool *zencode1, bool *ans, int len)
{
	int next = 0;
	// for(int i=0;i<len;i++){
	//     cout<<zencode0[i];
	// }cout<<endl;
	// for(int i=0;i<len;i++){
	//     cout<<zencode1[i];
	// }cout<<endl;

	for (int i = len - 1; i >= 0; i--)
	{
		int val = zencode0[i] + zencode1[i] + next;
		if (val == 2)
		{
			next = 1;
			ans[i] = 0;
		}
		else if (val == 1)
		{
			next = 0;
			ans[i] = 1;
		}
		else if (val == 3)
		{
			next = 1;
			ans[i] = 1;
		}
		else
		{
			next = 0;
			ans[i] = 0;
		}
	}
	if (next == 1)
	{
		ans[-1] = next;
	}
	// for(int i=0;i<len;i++){
	//     cout<<ans[i];
	// }cout<<endl;
}
float probeCDF(bool *zencode0, bool *zencode1, int diglen, MADENet *Mnet, Query Qi, int depth)
{
	bool *mid = new bool[diglen + 10];
	float cdfl = point2cdfest(Mnet, Qi, zencode0);
	float cdfu = point2cdfest(Mnet, Qi, zencode1);

	if (cdfl > cdfu)
	{
		return 0;
	}
	if (depth == 0)
	{
		// cout << "Dep" << depth << " L:" << cdfl << " R:" << cdfu <<" Delta:" <<   cdfu - cdfl<< endl;
		return cdfu - cdfl;
	}

	if ((cdfu - cdfl) < 0.00001)
	{
		return cdfu - cdfl;
	}

	mid[0] = 0;
	ZADD(zencode0, zencode1, &mid[1], diglen);
	// cout<<"MIN:";
	// for(int i=0;i<*tolbits;i++){
	//     cout<<zencode0[i];
	// }cout<<endl;
	// cout<<"MAX:";
	// for(int i=0;i<*tolbits;i++){
	//     cout<<zencode1[i];
	// }cout<<endl;
	// cout<<"MID:";
	// for(int i=0;i<*tolbits;i++){
	//     cout<<mid[i];
	// }cout<<endl;
	float cdfm = point2cdfest(Mnet, Qi, mid);
	// cout << "Dep" << depth << " L:" << cdfl << " R:" << cdfu << " mid:" << cdfm << endl;
	// ZTuple *Zt = makeZT(mid,diglen);
	bool flag = inboxZ(Qi, mid);
	// cout<< inboxZ(Qi,zencode0) <<" "<< inboxZ(Qi,zencode1)<<" "<<inboxZ(Qi,mid);
	// exit(1);
	// delete(Zt);
	// cout << "flag" << flag << endl;
	if (flag == true)
	{
		// cout<<"inbox"<<endl;
		// bool lm[120];
		// bool mu[120];
		// lm[0] = 0;
		// mu[0] = 0;
		// ZADD(zencode0, mid, &lm[1], diglen);
		// ZADD(zencode1, mid, &mu[1], diglen);
		float l = probeCDF(zencode0, mid, diglen, Mnet, Qi, depth - 1);
		float u = probeCDF(mid, zencode1, diglen, Mnet, Qi, depth - 1);
		return l + u;
	}
	else
	{
		// cout<<"ob"<<endl;
		bool minb[120];
		bool maxb[120];
		bool midb[120];
		bool tmpbmi[120];
		bool tmpmai[120];
		for (int i = 0; i < diglen; i++)
		{
			minb[i] = zencode0[i];
			maxb[i] = zencode1[i];
			midb[i] = mid[i];
		}
		bool *bigmin = getBIGMIN(minb, maxb, midb, diglen, Qi.binaryLength, Qi.columnNumber);
		bool *litmax = getLITMAX(minb, maxb, midb, diglen, Qi.binaryLength, Qi.columnNumber);
		for (int i = 0; i < diglen; i++)
		{

			tmpbmi[i] = bigmin[i];
			tmpmai[i] = litmax[i];
		}
		// tmpbmi[0] = 0;
		// tmpmai[0] = 0;
		// cout << cdfl << " " << point2cdfest(Mnet, Qi, tmpmai) << " " << point2cdfest(Mnet, Qi, tmpbmi) << " " << cdfu << endl;
		float ret = 0;
		ret += probeCDF(zencode0, tmpmai, diglen, Mnet, Qi, depth - 1);
		ret += probeCDF(tmpbmi, zencode1, diglen, Mnet, Qi, depth - 1);
		// cout<<abs( cdfl - point2cdfest(M,Qi,tmpmai)  )<<endl;
		// if(abs( cdfl - point2cdfest(M,Qi,tmpmai)  )>= 0.000001){
		//     ret += probeCDF(zencode0,tmpbmi,diglen,M,Qi,depth-1);
		// }
		// if(abs(cdfu -point2cdfest(M,Qi,tmpbmi)) >=0.000001){
		//     ret+= probeCDF(tmpmai,zencode1,diglen,M,Qi,depth-1);
		// }
		if (ret < 0)
		{
			return 0;
		}
		return ret;
		// exit(1);
	}
	// cout << cdfm << endl;
}

double preciseCDFGet(bool *zencode0, int diglen, CardIndex *C, Query Qi)
{
	ZTuple *ZT0 = makeZT(zencode0, *tolbits);
	int belongl = getBelongNum(C, ZT0);
	int flag = C->trans->Flag[belongl];
	if (flag == 0)
	{ // �����ܳ���split
		BPlusNode *subN = C->trans->transferLayer[belongl];
		int preN = linearNodeSearchPre(subN, ZT0);
		return (preN + (double)0.0) / C->Mnet->zdr;
	}
	else
	{
		KeyType SepKey = C->trans->transferLayer[belongl]->Key[0];
		if (SepKey->z32 < ZT0->z32) // this Node
		{
			BPlusNode *subN = C->trans->transferLayer[belongl];
			int preN = linearNodeSearchPre(subN, ZT0);
			return (preN + (double)0.0) / C->Mnet->zdr;
		}
		else
		{
			if (belongl == 0)
			{
				return 0.0;
			}
			BPlusNode *subN = C->trans->transferLayer[belongl - 1];
			int preN = linearNodeSearchPre(subN, ZT0);
			return (preN + (double)0.0) / C->Mnet->zdr;
		}
	}
}

bool GLBZEnc0[200];
bool GLBZEnc1[200];

float probeCDFPrecise(bool *zencode0, bool *zencode1, int diglen, CardIndex *C, Query Qi, int depth, int firstcall = 0)
{
	bool *mid = new bool[diglen + 10];
	mid[0] = 0;
	ZADD(zencode0, zencode1, &mid[1], diglen); // �Զ��������-1 slot
	// cout << "RightD:";
	// for (int i = 0; i < *tolbits; i++) {
	//	cout << zencode1[i];
	// }cout << endl;
	// cout << "LeftUp:";
	// for (int i = 0; i < *tolbits; i++) {
	//	cout << zencode0[i];
	// }cout << endl;
	// cout << "MIDVAL:";
	// for (int i = 0; i < *tolbits; i++) {
	//	cout << mid[i];
	// }cout << endl;
	double cdfl, cdfu;
	if (depth < 10)
	{
		cdfl = preciseCDFGet(zencode0, diglen, C, Qi);
		cdfu = preciseCDFGet(zencode1, diglen, C, Qi);
	}
	else
	{
		cdfl = point2cdfest(C->Mnet, Qi, zencode0);
		cdfu = point2cdfest(C->Mnet, Qi, zencode1);
	}
	bool flag1 = inboxZ(Qi, mid);
	if (firstcall)
	{

		if ((cdfu - cdfl) < bEST)//earlt stop
		{
			return cdfu - cdfl;
		}
	}
	// cout << "lm" << inboxZ(Qi, zencode0);
	// cout << "rm" << inboxZ(Qi, zencode1);
	/*cout << "In layer " << depth <<" ";
	cout << cdfl << " " << cdfu << endl;
	cout << inboxZ(Qi, zencode0) << " " << inboxZ(Qi, mid) << " " << inboxZ(Qi, zencode1) << endl;*/
	if (cdfl > cdfu)
	{
		return 0;
	}
	if (depth == 0 || (cdfu - cdfl) < 1e-4)
	{
		if (flag1)
		{
			return cdfu - cdfl;
		}
		bool flag2 = inboxZ(Qi, zencode0);
		bool flag3 = inboxZ(Qi, zencode1);
		if (flag2 + flag3 == 2)
		{
			return (cdfu - cdfl) / 2;
		}
		return 0;
	}
	// cout << flag2<<" " << flag1 << " " << flag3 << endl;
	//  exit(1);
	//  delete(Zt);
	//  cout << "flag" << flag << endl;
	if (flag1 == true)
	{
		// cout<<"inbox"<<endl;
		//  bool lm[120];
		//  bool mu[120];
		//  lm[0] = 0;
		//  mu[0] = 0;
		//  ZADD(zencode0, mid, &lm[1], diglen);
		//  ZADD(zencode1, mid, &mu[1], diglen);
		float l = probeCDFPrecise(zencode0, mid, diglen, C, Qi, depth - 1);
		if (l > bEST)
		{
			return l;
		}
		float u = probeCDFPrecise(mid, zencode1, diglen, C, Qi, depth - 1);
		// cout << "Out L:" << depth << l<<" "<<u<<endl;
		return l + u;
	}
	else
	{
		// cout<<"ob"<<endl;
		bool tmpbmi[120];
		bool tmpmai[120];
		bool *litmax = getLITMAXF(GLBZEnc0, GLBZEnc1, mid, *tolbits, Qi.binaryLength, Qi.columnNumber);
		bool *bigmin = getBIGMINF(GLBZEnc0, GLBZEnc1, mid, *tolbits, Qi.binaryLength, Qi.columnNumber);
		/*cout << "BIGMIN:";
		for (int i = 0; i < *tolbits; i++) {
			cout << bigmin[i];
		}cout << endl;
		cout << "MIDVAL:";
		for (int i = 0; i < *tolbits; i++) {
			cout << mid[i];
		}cout << endl;
		cout << "LITMAX:";
		for (int i = 0; i < *tolbits; i++) {
			cout << litmax[i];
		}cout << endl;*/

		// cout << "Cking litmax" << endl;
		// bool flagX = inboxZ(Qi, mid);
		// cout << "Cking bigmin" << endl;
		// bool flagX2 = 1;// inboxZ(Qi, minb);
		// bool flagY2 = 1;// inboxZ(Qi, maxb);
		// cout <<flagX2<<" " << flagZ << " "  << " " << flagY <<" " <<flagY2<< endl;
		memcpy(tmpbmi, bigmin, diglen);
		memcpy(tmpmai, litmax, diglen);

		// for (int i = 0; i < diglen; i++)
		//{
		//	tmpbmi[i] = bigmin[i];
		//	tmpmai[i] = litmax[i];
		// }
		//  tmpbmi[0] = 0;
		//  tmpmai[0] = 0;
		// cout << cdfl << " " << preciseCDFGet( tmpmai,diglen,C,Qi) << " " << preciseCDFGet( tmpbmi, diglen, C, Qi) << " " << cdfu << endl;
		float ret = 0;
		float leftCDF = probeCDFPrecise(zencode0, tmpmai, diglen, C, Qi, depth - 1);
		if (leftCDF > bEST)
		{
			return leftCDF;
		}
		float rightCDF = probeCDFPrecise(tmpbmi, zencode1, diglen, C, Qi, depth - 1);
		// cout << "Layer: " << depth << " Left:" << leftCDF << " Right:" << rightCDF << endl;
		ret += (leftCDF + rightCDF);
		// cout << "Out L:" << depth <<" " << leftCDF << " " << rightCDF << endl;
		//  cout<<abs( cdfl - point2cdfest(M,Qi,tmpmai)  )<<endl;
		//  if(abs( cdfl - point2cdfest(M,Qi,tmpmai)  )>= 0.000001){
		//      ret += probeCDF(zencode0,tmpbmi,diglen,M,Qi,depth-1);
		//  }
		//  if(abs(cdfu -point2cdfest(M,Qi,tmpbmi)) >=0.000001){
		//      ret+= probeCDF(tmpmai,zencode1,diglen,M,Qi,depth-1);
		//  }
		if (ret < 0)
		{
			return 0;
		}
		return ret;
		// exit(1);
	}
	// cout << cdfm << endl;
}
float probeParrel(bool *zencode0, bool *zencode1, int diglen, CardIndex *C, Query Qi, int depth)
{
	depth--;
	const int numThreads = 2;					  // �����߳���
	std::vector<std::thread> threads(numThreads); // �����߳�����
	std::vector<float> results(numThreads);		  // ��������ÿ���̷߳���ֵ������
	bool *mid = new bool[diglen + 10];
	mid[0] = 0;
	ZADD(zencode0, zencode1, &mid[1], diglen); // �Զ��������-1 slot
	if (inboxZ(Qi, mid))
	{
		threads[0] = std::thread([&](int threadIndex)
								 {
			// ÿ���̵߳��ú��������淵��ֵ
			results[threadIndex] = probeCDFPrecise(zencode0, mid, diglen, C, Qi, depth, 0); },
								 0);

		threads[1] = std::thread([&](int threadIndex)
								 {
			// ÿ���̵߳��ú��������淵��ֵ
			results[threadIndex] = probeCDFPrecise(mid, zencode1, diglen, C, Qi, depth, 0); },
								 1);

		// �ȴ������߳����
		for (auto &thread : threads)
		{
			thread.join();
		}
		// �������з���ֵ���ܺ�
		float sum = std::accumulate(results.begin(), results.end(), 0.0f);
		return sum;
	}
	else
	{
		bool minb[120];
		bool maxb[120];
		bool midb[120];
		bool tmpbmi[120];
		bool tmpmai[120];
		for (int i = 0; i < *tolbits; i++)
		{
			minb[i] = zencode0[i];
			maxb[i] = zencode1[i];
			midb[i] = mid[i];
		}
		bool *litmax = getLITMAXF(minb, maxb, midb, *tolbits, Qi.binaryLength, Qi.columnNumber);
		// cout << diglen<<" "<<Qi.columnNumber << endl;
		for (int i = 0; i < *tolbits; i++)
		{
			minb[i] = zencode0[i];
			maxb[i] = zencode1[i];
			midb[i] = mid[i];
		}
		bool *bigmin = getBIGMINF(minb, maxb, midb, *tolbits, Qi.binaryLength, Qi.columnNumber);

		threads[0] = std::thread([&](int threadIndex)
								 {
			// ÿ���̵߳��ú��������淵��ֵ
			results[threadIndex] = probeCDFPrecise(zencode0, litmax, diglen, C, Qi, depth, 0); },
								 0);

		threads[1] = std::thread([&](int threadIndex)
								 {
			// ÿ���̵߳��ú��������淵��ֵ
			results[threadIndex] = probeCDFPrecise(bigmin, zencode1, diglen, C, Qi, depth, 0); },
								 1);

		// �ȴ������߳����
		for (auto &thread : threads)
		{
			thread.join();
		}
		// �������з���ֵ���ܺ�
		float sum = std::accumulate(results.begin(), results.end(), 0.0f);
		return sum;
	}
}
float probeParrelN(bool *zencode0, bool *zencode1, int diglen, CardIndex *C, Query Qi, int depth)
{
	depth -= 2;
	const int numThreads = 4;					  // �����߳���
	std::vector<std::thread> threads(numThreads); // �����߳�����
	std::vector<float> results(numThreads);		  // ��������ÿ���̷߳���ֵ������
	bool *mid = new bool[diglen + 10];
	bool *m14 = new bool[diglen + 10];
	bool *m34 = new bool[diglen + 10];

	mid[0] = 0;
	m14[0] = 0;
	m34[0] = 0;

	ZADD(zencode0, zencode1, &mid[1], diglen); // �Զ��������-1 slot
	ZADD(zencode0, mid, &m14[1], diglen);
	ZADD(zencode1, mid, &m34[1], diglen);
	bool *t0L = zencode0;
	bool *t0r;
	bool *t1L;
	bool *t1r;
	bool *t2L;
	bool *t2r;
	bool *t3L;
	bool *t3r = zencode1;
	if (inboxZ(Qi, m14))
	{
		t0r = m14;
		t1L = m14;
	}
	else
	{
		bool *litmax = getLITMAXF(zencode0, zencode1, m14, *tolbits, Qi.binaryLength, Qi.columnNumber);
		bool *bigmin = getBIGMINF(zencode0, zencode1, m14, *tolbits, Qi.binaryLength, Qi.columnNumber);
		t0r = new bool[120];
		t1L = new bool[120];
		memcpy(t0r, litmax, *tolbits);
		memcpy(t1L, bigmin, *tolbits);
	}
	if (inboxZ(Qi, mid))
	{
		t1r = mid;
		t2L = mid;
	}
	else
	{
		bool *litmax = getLITMAXF(zencode0, zencode1, mid, *tolbits, Qi.binaryLength, Qi.columnNumber);
		bool *bigmin = getBIGMINF(zencode0, zencode1, mid, *tolbits, Qi.binaryLength, Qi.columnNumber);
		t1r = new bool[120];
		t2L = new bool[120];
		memcpy(t1r, litmax, *tolbits);
		memcpy(t2L, bigmin, *tolbits);
	}
	if (inboxZ(Qi, m34))
	{
		t2r = m34;
		t3L = m34;
	}
	else
	{
		bool *litmax = getLITMAXF(zencode0, zencode1, m34, *tolbits, Qi.binaryLength, Qi.columnNumber);
		bool *bigmin = getBIGMINF(zencode0, zencode1, m34, *tolbits, Qi.binaryLength, Qi.columnNumber);
		t2r = new bool[120];
		t3L = new bool[120];
		memcpy(t2r, litmax, *tolbits);
		memcpy(t3L, bigmin, *tolbits);
	}
	threads[0] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = probeCDFPrecise(t0L, t0r, diglen, C, Qi, depth, 0); },
							 0);

	threads[1] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = probeCDFPrecise(t1L, t1r, diglen, C, Qi, depth, 0); },
							 1);

	threads[2] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = probeCDFPrecise(t2L, t2r, diglen, C, Qi, depth, 0); },
							 2);

	threads[3] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = probeCDFPrecise(t3L, t3r, diglen, C, Qi, depth, 0); },
							 3);
	for (auto &thread : threads)
	{
		thread.join();
	}
	// �������з���ֵ���ܺ�
	float sum = std::accumulate(results.begin(), results.end(), 0.0f);
	return sum;
}

int rq[20];
void rangeThreadProcess(int idx, int belong, int end, CardIndex *C, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1)
{

	int estcard = 0;
	BPlusNode *past = NULL;
	for (int xi = belong; xi < end; xi = xi)
	{
		// cout<<"searching:"<<xi<<endl;
		BPlusNode *ptr = C->trans->transferLayer[xi];
		if (ptr == NULL || ptr == past)
		{
			xi++;
			continue;
		}
		else
		{
			int searchC = rangeQueryExceuteF(ptr, qi, zencode0, zencode1, ZT0, ZT1);
			estcard += searchC;
			if (searchC == 0)
			{
				// cout<<"BMnf"<<endl;
				bool *bigmin = getBIGMIN(ZT0->bin, ZT1->bin, lastN->Key[lastN->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
				ZTuple *ZTX = makeZT(bigmin, *tolbits);
				int belongx = getBelongNum(C, ZTX);
				if (belongx > xi)
				{
					// cout<<"skip"<<xi<<" "<<belongx<<endl;
					xi = belongx;
					past = ptr;
					continue;
				}
			}
			past = ptr;
			xi++;
		}
	}
	rq[idx] = estcard;
}
bool zTcmpL(ZTuple *ZT0, ZTuple *ZT1, int cmplen)
{
	// if 0< 1:True
	for (int i = 0; i < cmplen; i++)
	{
		if (ZT0->bin[i] < ZT1->bin[i])
		{
			return true;
		}
		else if (ZT0->bin[i] > ZT1->bin[i])
		{
			return false;
		}
		else
		{
			continue;
		}
	}
	return false;
}
long long purescan = 0;
int singleRangeQ(CardIndex *C, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1)
{
	/*for (int i = 0; i < MAXL1CHILD; i++) {
		cout << i << " " << C->trans->Flag[i] << endl;
	}
	exit(1);*/
	BPlusNode *pre = CardIndexLeafGet(C, ZT0);
	BPlusNode *post = CardIndexLeafGet(C, ZT1);
	int estcard = 0;
	long long scanned = 0;
	long long missed = 0;
	long long eqmiss = 0;
	int jumped = 0;
	int tolBlks = 0;
	typedef std::chrono::high_resolution_clock Clock;
	long long scanTime = 0;
	long long bmTime = 0;
	long long jumpFail = 0;
	long long pqTime = 0;
	for (Position p0 = pre; p0 != NULL && p0 != post; p0 = p0->Next)
	{
		tolBlks++;
	}
	int processBlk = 0;
	for (Position tmp = pre; (tmp != NULL) && (tmp->Key[0]->z32 <= post->Key[0]->z32); tmp = tmp)
	{
		auto scanBeg = Clock::now();
		processBlk += 1;
		lastN = tmp;
		int flag = 0;
		for (int ti = 0; ti < tmp->KeyNum; ti++)
		{
			scanned += 1;
			if (inbox(&qi, tmp->Key[ti]))
			{
				estcard += 1;
				flag = 1;
			}
		}
		auto scanEnd = Clock::now();
		scanTime += std::chrono::duration_cast<std::chrono::nanoseconds>(scanEnd - scanBeg).count();

		if (flag == 0)
		{
			missed++;
			auto bigminS = Clock::now();
			bool *bigmin = getBIGMINF(&zencode0[1], &zencode1[1], tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
			auto bigminE = Clock::now();
			ZTuple *ZTX = makeZT(bigmin, *tolbits);
			Position pt = CardIndexLeafGet(C, ZTX);
			auto PointEx = Clock::now();
			bmTime += std::chrono::duration_cast<std::chrono::nanoseconds>(bigminE - bigminS).count();
			pqTime += std::chrono::duration_cast<std::chrono::nanoseconds>(PointEx - bigminE).count();
			if (pt == NULL)
			{
				tmp = tmp->Next;
			}
			if (pt->Key[0]->z32 <= tmp->Key[0]->z32) // Full cmp
			{
				jumpFail++;
				if (tmp->Key[0]->z32 == pt->Key[0]->z32)
				{
					eqmiss++;
				}
				// cout << pt->Key[0]->z32 <<" " << tmp->Key[0]->z32 <<" " << pt << " " << tmp<< endl;
				tmp = tmp->Next;
			}
			else
			{
				jumped++;
				tmp = pt;
			}
		}
		else
		{
			tmp = tmp->Next;
		}
	}
	// cout << "Scanned: " << scanned << " Missed: " << missed <<"JumpFail: " << jumpFail << " EQMISS:" << eqmiss << " Jumped :" << jumped << endl;
	// cout << "BLKBET:" << tolBlks << " Processblks:" << processBlk << endl;
	// cout << "BIGMIN_T: " << bmTime << "  SCANT:" << scanTime << " PQTime:" << pqTime << endl;
	purescan += scanTime;
	return estcard;
}
int searchBet(CardIndex *C, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1, BPlusNode *pre, BPlusNode *post)
{
	long long estcard = 0;
	if (post == NULL)
	{
		for (Position tmp = pre; (tmp != NULL); tmp = tmp)
		{
			lastN = tmp;
			int flag = 0;
			for (int ti = 0; ti < tmp->KeyNum; ti++)
			{
				if (inbox(&qi, tmp->Key[ti]))
				{
					estcard += 1;
					flag = 1;
				}
			}
			if (flag == 0)
			{
				bool *bigmin = getBIGMIN(&zencode0[1], &zencode1[1], tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
				ZTuple *ZTX = makeZT(bigmin, *tolbits);
				Position pt = CardIndexLeafGet(C, ZTX);
				if (pt == NULL)
				{
					tmp = tmp->Next;
				}
				if (pt->Key[0]->z32 <= tmp->Key[0]->z32) // Full cmp
				{
					tmp = tmp->Next;
				}
				else
				{
					tmp = pt;
				}
			}
			else
			{
				tmp = tmp->Next;
			}
		}
		return estcard;
	}
	/*cout << pre << " " << post << " " << endl;
	cout << pre->Key[0]->z32 << " " << post->Key[0]->z32 << endl;*/
	for (Position tmp = pre; (tmp != NULL) && (tmp->Key[0]->z32 <= post->Key[0]->z32); tmp = tmp)
	{
		lastN = tmp;
		int flag = 0;
		for (int ti = 0; ti < tmp->KeyNum; ti++)
		{
			if (inbox(&qi, tmp->Key[ti]))
			{
				estcard += 1;
				flag = 1;
			}
		}
		if (flag == 0)
		{
			bool *bigmin = getBIGMIN(&zencode0[1], &zencode1[1], tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
			ZTuple *ZTX = makeZT(bigmin, *tolbits);
			Position pt = CardIndexLeafGet(C, ZTX);
			if (pt == NULL)
			{
				tmp = tmp->Next;
			}
			if (pt->Key[0]->z32 <= tmp->Key[0]->z32) // Full cmp
			{
				tmp = tmp->Next;
			}
			else
			{
				tmp = pt;
			}
		}
		else
		{
			tmp = tmp->Next;
		}
	}
	return estcard;
}
int paraRangeQ(CardIndex *C, Query qi, bool *zencode0, bool *zencode1, ZTuple *ZT0, ZTuple *ZT1)
{
	// singleRangeQ�Ķ��߳�ʵ��
	BPlusNode *pre = CardIndexLeafGet(C, ZT0);
	BPlusNode *post = CardIndexLeafGet(C, ZT1);
	int numThreads = 16;
	int tolBlks = 0;
	for (Position p0 = pre; p0 != NULL && p0 != post; p0 = p0->Next)
	{
		tolBlks++;
	}
	if (tolBlks <= 256)
	{
		return singleRangeQ(C, qi, zencode0, zencode1, ZT0, ZT1);
	}
	// cout << tolBlks << endl;
	std::vector<std::thread> threads(numThreads); // �����߳�����
	std::vector<long long> results(numThreads);	  // ��������ÿ���̷߳���ֵ������
	int threadIdx = 0;
	// �����߳����
	BPlusNode *subBegin = NULL;
	int assignedBLK = 0;
	int blkperthead = tolBlks / numThreads;
	vector<vector<BPlusNode *>> ThreadPrepare;
	for (Position p0 = pre; p0 != NULL && p0 != post; p0 = p0->Next)
	{
		assignedBLK++;
		if (subBegin == NULL)
		{
			subBegin = p0;
		}
		if (assignedBLK > blkperthead)
		{
			assignedBLK = 0;
			ThreadPrepare.push_back({subBegin, p0});
			subBegin = NULL;
			threadIdx++;
		}
	}
	ThreadPrepare.push_back({subBegin, post});
	int idx = 0;
	threads[0] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[0][0], ThreadPrepare[0][1]); },
							 0);
	threads[1] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[1][0], ThreadPrepare[1][1]); },
							 1);
	threads[2] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[2][0], ThreadPrepare[2][1]); },
							 2);
	threads[3] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[3][0], ThreadPrepare[3][1]); },
							 3);

	threads[4] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[4][0], ThreadPrepare[4][1]); },
							 4);
	threads[5] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[5][0], ThreadPrepare[5][1]); },
							 5);
	threads[6] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[6][0], ThreadPrepare[6][1]); },
							 6);
	threads[7] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[7][0], ThreadPrepare[7][1]); },
							 7);

	threads[8] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[8][0], ThreadPrepare[8][1]); },
							 8);

	threads[9] = std::thread([&](int threadIndex)
							 {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[9][0], ThreadPrepare[9][1]); },
							 9);

	threads[10] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[10][0], ThreadPrepare[10][1]); },
							  10);

	threads[11] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[11][0], ThreadPrepare[11][1]); },
							  11);

	threads[12] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[12][0], ThreadPrepare[12][1]); },
							  12);

	threads[13] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[13][0], ThreadPrepare[13][1]); },
							  13);

	threads[14] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[14][0], ThreadPrepare[14][1]); },
							  14);

	threads[15] = std::thread([&](int threadIndex)
							  {
		// ÿ���̵߳��ú��������淵��ֵ
		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, ThreadPrepare[15][0], ThreadPrepare[15][1]); },
							  15);

	// for (auto x : ThreadPrepare) {
	//	cout << x[0]->Key[0]->z32 << " " << x[1]->Key[0]->z32 << endl;
	//	cout << x[0] << " " << x[1] << endl;
	//	cout << "T" << endl;
	//	threads[idx] = std::thread([&](int threadIndex) {
	//		// ÿ���̵߳��ú��������淵��ֵ
	//		results[threadIndex] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, x[0], x[1]);
	//		}, idx);
	//	idx++;
	// }
	// exit(1);
	// if (subBegin != NULL) {//IF not NULL:
	//	results[threadIdx] = searchBet(C, qi, zencode0, zencode1, ZT0, ZT1, subBegin, post);
	// }
	//
	//// �ȴ������߳����
	for (auto &thread : threads)
	{
		thread.join();
	}
	//// �������з���ֵ���ܺ�
	long long sum = std::accumulate(results.begin(), results.end(), 0.0f);
	return sum;
}
int LinearNodeSearch(BPlusNode *subN, ZTuple *ZT0)
{
	// �������Ժ���������������
	// �������<=ZT0������Idx
	int childIdx = -1;
	// cout<<ZT0->z32<<" "<<subN->Key[0]->z32<<endl;
	/*if (subN->Zmin != subN->Key[0]->z32) {
		cout << "werid" << endl;
		cout << subN->KeyNum << endl;
		cout << subN->Zmin << " " << subN->Zmax << endl;
		cout << subN->Key[0]->z32 << " " << subN->Key[subN->KeyNum-1]->z32 << endl;
		exit(1);
	}*/
	if (ZT0->z32 <= subN->Zmin)
	{

		if (ZT0->z32 == subN->Zmin)
		{
			for (int i = 1; i < subN->KeyNum; i++)
			{
				if (subN->Key[i]->z32 > ZT0->z32)
				{
					return i - 1;
				}
			}
			return subN->KeyNum - 1;
		}
		return 0;
	}

	if (subN->Zmin == subN->Zmax)
	{
		// z32ȫһ���ġ�����,���ǣ�ZT0->z32�����Ƕ���
		return subN->KeyNum - 1;
	}
	double X_norm = (ZT0->z32 - subN->Zmin) / (subN->Zmax - subN->Zmin + 0.0);

	for (int i = 0; i < subN->linNum; i++)
	{
		// cout<<i<<" "<<ZT0->z32<<" "<<subN->LinearX[i].x0<<endl;
		if (subN->LinearX[i].x0 > X_norm)
		{
			double x0 = subN->LinearX[i - 1].x0;
			double y0 = subN->LinearX[i - 1].y0;
			double kup = subN->LinearX[i - 1].kup;
			double kdown = subN->LinearX[i - 1].kdown;
			double delta_y_Up = kup * (X_norm - x0);
			double delta_y_Down = kdown * (X_norm - x0);
			// cout << kup << " " << kdown << " " << delta_y_Up << " " << delta_y_Down << endl;
			double predUp = delta_y_Up + y0;
			double predDown = delta_y_Down + y0;
			// cout << y0 << " " << predUp << " " << predDown << endl;
			if (predDown < y0)
			{
				predDown = y0;
			}
			for (int j = (int)predDown; j <= (predUp) && j < subN->KeyNum; j++)
			{
				/*if (ZT0->z32 == subN->Key[j]->z32)
				{
					return j;
				}*/
				// else
				if (subN->Key[j]->z32 <= ZT0->z32)
				{
					childIdx = j;
					// break;
				}
			}
		}
	}
	if (childIdx == -1)
	{
		// cout << "detect" << endl;
		int i = subN->linNum;
		double x0 = subN->LinearX[i - 1].x0;
		double y0 = subN->LinearX[i - 1].y0;
		double kup = subN->LinearX[i - 1].kup;
		double kdown = subN->LinearX[i - 1].kdown;
		double delta_y_Up = kup * (X_norm - x0);
		double delta_y_Down = kdown * (X_norm - x0);
		// cout << kup << " " << kdown << " " << delta_y_Up << " " << delta_y_Down << endl;
		double predUp = delta_y_Up + y0;
		double predDown = delta_y_Down + y0;
		/*long long x0 = subN->LinearX[i - 1].x0;
		long long y0 = subN->LinearX[i - 1].y0;
		float kup = subN->LinearX[i - 1].kup;
		float kdown = subN->LinearX[i - 1].kdown;
		float delta_y_Up = kup * (ZT0->z32 - x0);
		float delta_y_Down = kdown * (ZT0->z32 - x0);
		float predUp = delta_y_Up + y0;
		float predDown = delta_y_Down + y0;*/
		// cout<<predUp<<" "<<predDown<<endl;
		if (predDown < y0)
		{
			predDown = y0;
		}
		if (subN->Key[(int)subN->LinearX[i - 1].y0]->z32 == ZT0->z32)
		{
			childIdx = subN->LinearX[i - 1].y0;
		}
		else
		{
			for (int j = y0; j <= (predUp) && j < subN->KeyNum; j++)
			{
				if (ZT0->z32 > subN->Key[j]->z32)
				{
					childIdx = j;
				}
				else if (ZT0->z32 == subN->Key[j]->z32)
				{
					childIdx = j;
					continue;
					// return j;
				}
				else
				{
					break;
				}
			}
		}
	}

	return childIdx;
}
void testCorrectPointQuery(BPlusNode *LinkEDList, CardIndex *CI)
{
	typedef std::chrono::high_resolution_clock Clock;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
}
BPlusNode *linearLeafSearch(BPlusNode *subN, ZTuple *ZT0)
{
	// �������Ժ���������������
	// �������<=ZT0��Ҷ�ڵ�ָ��
	if (subN->Children[0] == NULL)
	{
		return subN;
	}
	int childIdx = LinearNodeSearch(subN, ZT0);
	if (childIdx < 0)
	{
		childIdx = 0;
	}
	else if (childIdx > subN->KeyNum)
	{
		childIdx = subN->KeyNum - 1;
	}
	BPlusNode *child = subN->Children[childIdx];
	return linearLeafSearch(child, ZT0);
}

bool showLinearNodePerformance(BPlusNode *subN, ZTuple *ZT0)
{
	bool leafFlg = 0;
	if (subN->Children[0] == NULL)
	{
		leafFlg = 1;
	}
	int childIdx = LinearNodeSearch(subN, ZT0);
	// cout<<childIdx<<" "<<subN->KeyNum<<endl;
	if (leafFlg)
	{
		// cout<<subN->Key[childIdx]->z32 <<" " <<ZT0->z32<<endl;
		if (subN->Key[childIdx]->z32 == ZT0->z32)
		{
			// cout << "Find" << endl;
			return true;
		}
		// cout << "NF" << endl;
		// exit(1);
		return false;
	}
	else
	{
		return showLinearNodePerformance(subN->Children[childIdx], ZT0);
	}
}
int linearNodeSearchPre(BPlusNode *subN, ZTuple *ZT0)
{
	bool leafFlg = 0;
	if (subN == NULL)
	{
		return NULL;
	}
	if (subN->Children[0] == NULL)
	{
		leafFlg = 1;
	}
	int childIdx = LinearNodeSearch(subN, ZT0);
	if (childIdx < 0)
	{
		childIdx = 0;
	}
	if (childIdx >= subN->KeyNum)
	{
		childIdx = subN->KeyNum - 1;
	}
	if (leafFlg)
	{
		if (subN->Key[childIdx]->z32 == ZT0->z32)
		{
			return childIdx + subN->preNum;
		}
		return subN->preNum;
	}
	else
	{
		return linearNodeSearchPre(subN->Children[childIdx], ZT0);
	}
}
bool showSubNodePerformance(BPlusNode *subN, ZTuple *ZT0)
{
	BPlusNode *leaf = recrusiveFind(subN, ZT0);
	int flag = 0;
	for (int j = 0; j < leaf->KeyNum; j++)
	{
		// cout<<p0->Key[j]->z32<<" "<<ZT0->z32<<endl;
		if (leaf->Key[j]->z32 == ZT0->z32)
		{
			// cout << "Find" << endl;
			flag = 1;
			return true;
			break;
		}

		// cout<<p0->Key[j]->z32<<" "<<ZT0->z32<<endl;
	}
	if (flag == 0)
	{
		// cout << "nf" << endl;
		return false;
		// exit(1);
	}
	return true;
}

void testCardIndexRangeQuery(CardIndex *C, string queryfilepath)
{
	Querys *qs = readQueryFile(queryfilepath);
	cout << "Query loaded: "
		 << " " << queryfilepath << endl;
	ofs << queryfilepath << endl;
	vector<float> ABSL;
	typedef std::chrono::high_resolution_clock Clock;
	long long timesum = 0, NonleafCnt = 0;
	long long queryestTime = 0;
	cout << "Query Num:" << qs->queryNumber << endl;
	int up = qs->queryNumber;
	maxlen = 0;
	for (int i = 0; i < qs->Qs[0].columnNumber; i += 1)
	{
		if (qs->Qs[0].binaryLength[i] > maxlen)
		{
			maxlen = qs->Qs[0].binaryLength[i];
		}
	}
	buildColPattern(*tolbits, qs->Qs[0].binaryLength, qs->Qs[0].columnNumber);
	int estCard;
	long long processTime = 0;
	for (int i = 0; i < up; i++)
	{
		// if (i == 0)
		//{
		//	i = 1;
		// }
		// else {
		//	exit(1);
		// }
		estCard = 0;
		Query qi = qs->Qs[i];
		// cout << "QID:" << i;
		bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
		ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
		bool *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
		ZTuple *ZT1 = makeZT(&zencode1[1], *tolbits);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		estCard = singleRangeQ(C, qi, zencode0, zencode1, ZT0, ZT1);
		auto t4 = Clock::now(); // ��ʱ��ʼ
		// cout << "QID:" << i;
		// ofs
		cout << "QID:" << i << "EstCard: " << estCard << " TrueCard: " << qid2TrueNumber[i] << "Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() << std::endl;
		cout << "EstCard: " << estCard << " TrueCard: " << qid2TrueNumber[i];
		cout << "Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() << std::endl;
		cout << "------------------------\n";
		processTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
	}
	cout << "Avg:" << processTime / up << " " << purescan / up << endl;
	ofs << "Avg:" << processTime / up << " " << purescan / up << endl;
	// exit(1);
}

int CardIndexRangeExceute(CardIndex *C, Query qi)
{
	int estCard = 0;
	bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
	ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
	bool *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
	ZTuple *ZT1 = makeZT(&zencode1[1], *tolbits);
	int subModelBelong0 = getBelongNum(C, ZT0);
	bool flag0 = C->trans->Flag[subModelBelong0];
	int subModelBelong1 = getBelongNum(C, ZT1);
	if (subModelBelong1 == MAXL1CHILD)
	{
		subModelBelong1--;
	}
	// bool flag1 = C->trans->Flag[subModelBelong1];
	// �����ܳ������Ƶ�������Ҳ಻����
	BPlusNode *subLeft;
	// cout<<subModelBelong0<<" "<<subModelBelong1<<endl;
	BPlusNode *subRight = C->trans->transferLayer[subModelBelong1];
	if (flag0 == 0)
	{
		subLeft = C->trans->transferLayer[subModelBelong0];
	}
	else
	{
		KeyType SepKey = C->trans->transferLayer[subModelBelong0]->Key[0];
		// cout<<(unsigned)SepKey->z32 <<" "<< (unsigned)ZT0->z32<<endl;
		if ((unsigned)SepKey->z32 < (unsigned)ZT0->z32) // this Node
		{
			subLeft = C->trans->transferLayer[subModelBelong0];
		}
		else
		{
			if (subModelBelong0 == 0) // not find, too small Z value
			{
				subLeft = C->Head;
			}
			else
			{
				subLeft = C->trans->transferLayer[subModelBelong0 - 1];
			}
		}
	}
	Position Ps0 = recrusiveFind(subLeft, ZT0);
	Position Ps1 = recrusiveFind(subRight, ZT1);
	// cout<<Ps0->Key[0]->z32<<" "<<Ps1->Key[0]->z32<<endl;
	for (Position tmp = Ps0; (tmp != Ps1->Next) && (tmp != NULL); tmp = tmp)
	{
		int flag = 0;
		for (int ti = 0; ti < tmp->KeyNum; ti++)
		{
			if (inbox(&qi, tmp->Key[ti]))
			{
				estCard += 1;
				flag = 1;
			}
		}
		// tmp = tmp->Next;
		// continue;
		if (flag == 0)
		{
			bool *bigmin = getBIGMIN(ZT0->bin, ZT1->bin, tmp->Key[tmp->KeyNum - 1]->bin, *tolbits, qi.binaryLength, qi.columnNumber);
			ZTuple *ZTX = makeZT(bigmin, *tolbits);
			int subModelBelongX = getBelongNum(C, ZTX);
			bool flagX = C->trans->Flag[subModelBelongX];
			Position nextPos;
			if (flagX == 0)
			{
				nextPos = C->trans->transferLayer[subModelBelongX];
			}
			else
			{
				KeyType SepKeyX = C->trans->transferLayer[subModelBelongX]->Key[0];
				if ((unsigned)SepKeyX->z32 < (unsigned)ZTX->z32) // this Node
				{
					nextPos = C->trans->transferLayer[subModelBelongX];
				}
				else
				{
					if (subModelBelongX == 0) // not find, too small Z value
					{
						nextPos = C->Head;
					}
					else
					{
						nextPos = C->trans->transferLayer[subModelBelongX - 1];
					}
				}
			}
			Position pt = recrusiveFind(nextPos, ZTX);
			if ((unsigned)pt->Key[0]->z32 <= (unsigned)tmp->Key[0]->z32)
			{
				tmp = tmp->Next;
			}
			else
			{
				tmp = pt;
			}
		}
		else
		{
			tmp = tmp->Next;
		}
	}
	return estCard;
}
BPlusNode *CardIndexLeafGet(CardIndex *C, ZTuple *ZT0)
{
	int subModelBelong = getBelongNum(C, ZT0);
	bool flag = C->trans->Flag[subModelBelong];
	// cout << subModelBelong << " " << flag << endl;
	if (flag == 0)
	{
		BPlusNode *subN = C->trans->transferLayer[subModelBelong];
		// cout << subModelBelong << " " << subN << endl;
		return linearLeafSearch(subN, ZT0);
	}
	else
	{
		KeyType SepKey = C->trans->transferLayer[subModelBelong]->Key[0];
		if (SepKey->z32 < ZT0->z32) // this Node
		{
			// cout << "Ts node" << endl;
			BPlusNode *subN = C->trans->transferLayer[subModelBelong];
			return linearLeafSearch(subN, ZT0);
		}
		else
		{ // previous Node
			// cout << "Pre" << endl;
			if (subModelBelong == 0)
			{
				BPlusNode *subN = C->trans->transferLayer[0];
				return linearLeafSearch(subN, ZT0);
			}
			else
			{
				BPlusNode *subN = C->trans->transferLayer[subModelBelong - 1];
				return linearLeafSearch(subN, ZT0);
			}
		}
	}
}
void testCardIndexPointQuery(CardIndex *C, string queryfilepath)
{
	Querys *qs = readQueryFile(queryfilepath);
	cout << "Query loaded" << endl;
	vector<float> ABSL;
	typedef std::chrono::high_resolution_clock Clock;
	long long timesum = 0, NonleafCnt = 0;
	long long queryestTime = 0;
	cout << "Query Num:" << qs->queryNumber << endl;
	int up = qs->queryNumber;
	// up = 100;

	for (int j = 0; j <= 100; j++)
	{
		for (int i = 0; i < up; i++)
		{
			// i = 5;
			Query qi = qs->Qs[i];
			// cout << "QID:" << i<<" :";
			bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
			ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
			auto t0 = Clock::now(); // ��ʱ��ʼ;
			int subModelBelong = getBelongNum(C, ZT0);

			// auto t1 = Clock::now(); // ��ʱ��ʼ;
			// timesum += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
			// continue;
			// cout << subModelBelong << endl;
			// cout << "Reached CKPT" << endl;
			bool flag = C->trans->Flag[subModelBelong];
			// cout << "Flag:" << flag << endl;
			if (flag == 0)
			{ // �����ܳ���split
				BPlusNode *subN = C->trans->transferLayer[subModelBelong];
				showLinearNodePerformance(subN, ZT0);
				// showSubNodePerformance(subN, ZT0);
			}
			else
			{
				KeyType SepKey = C->trans->transferLayer[subModelBelong]->Key[0];
				// cout<<(unsigned)SepKey->z32 <<" "<< (unsigned)ZT0->z32<<endl;
				if ((unsigned)SepKey->z32 < (unsigned)ZT0->z32) // this Node
				{
					BPlusNode *subN = C->trans->transferLayer[subModelBelong];
					bool fla = showLinearNodePerformance(subN, ZT0);
					// bool fla = showSubNodePerformance(subN, ZT0);
					if (fla == false)
					{
						// cout << "NF!!" << endl;
						// exit(1);
					}
				}
				else if ((unsigned)SepKey->z32 > (unsigned)ZT0->z32)
				{
					// cout<<"Pre"<<endl;
					if (subModelBelong == 0)
					{
						// cout << "0nf" << endl;
						// exit(1);
					}
					BPlusNode *subN = C->trans->transferLayer[subModelBelong - 1];
					// cout << "Pre: " << subModelBelong - 1 << " " << subN << endl;
					bool fla = showLinearNodePerformance(subN, ZT0);
					// bool fla = showSubNodePerformance(subN, ZT0);
					if (fla == false)
					{
						// cout << "NF!!" << endl;
						// exit(1);
					}
				}
				else
				{ // ��ʵ�������鷳�����ȼ��ǰһ����ȷ��������ġ�
					int a = 1;
					// cout << "find" << endl;
					// BPlusNode *subN = C->trans->transferLayer[subModelBelong-1];
					// bool previousAns= showSubNodePerformance(subN, ZT0);
				}
			}
			auto t1 = Clock::now(); // ��ʱ��ʼ;
			timesum += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
			// exit(1);
		}
	}
	cout << "ex time:" << timesum / (0.0 + qs->queryNumber * 100);
}
void deepCopyQuery(Query *src, Query *dst)
{

	dst->columnNumber = src->columnNumber;
	dst->binaryLength = new int[dst->columnNumber];
	for (int i = 0; i < dst->columnNumber; i++)
	{
		dst->binaryLength[i] = src->binaryLength[i];
	}

	dst->queryid = src->queryid;
	dst->leftupBound = new long long[dst->columnNumber];
	dst->rightdownBound = new long long[dst->columnNumber];
	for (int i = 0; i < dst->columnNumber; i++)
	{
		dst->leftupBound[i] = src->leftupBound[i];
		dst->rightdownBound[i] = src->rightdownBound[i];
	}
}
void deepCopyMade(MADENet *src, MADENet *dst)
{
	dst->zdr = src->zdr;
	dst->zdc = src->zdc;
	dst->connectlen = src->connectlen;
	dst->diglen = src->diglen;
	dst->leafnums = src->leafnums;
	dst->fc1w = new float[src->diglen * src->diglen];
	dst->fc2w = new float[src->diglen * src->diglen];
	dst->fc1b = new float[src->diglen];
	dst->fc2b = new float[src->diglen];
	int bittol = src->diglen;
	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			dst->fc1w[i * bittol + j] = src->fc1w[i * bittol + j];
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		dst->fc1b[i] = src->fc1b[i];
	}
	for (int i = 0; i < bittol; i++)
	{
		for (int j = 0; j < bittol; j++)
		{
			dst->fc2w[i * bittol + j] = src->fc2w[i * bittol + j];
		}
	}
	for (int i = 0; i < bittol; i++)
	{
		dst->fc2b[i] = src->fc2b[i];
	}
}
void testCardPerformance(CardIndex *C, string queryfilepath, int ProbeTest = -1)
{
	vector<float> pdist;
	Querys *qs = readQueryFile(queryfilepath);
	// cout << "Query loaded" << endl;
	int sampleN = 2000;
	// cout << C->Mnet->zdr << endl;
	float p50, p95, p90, p99;
	vector<float> ABSL;
	typedef std::chrono::high_resolution_clock Clock;
	long long timesum = 0, NonleafCnt = 0;
	long long CDFEstTime = 0;
	long long queryestTime = 0;
	// cout << qs->queryNumber << endl;
	int up = qs->queryNumber;
	// prepare
	maxlen = 0;
	*tolbits = 0;
	for (int i = 0; i < qs->Qs[0].columnNumber; i += 1)
	{
		(*tolbits) += qs->Qs[0].binaryLength[i];
		if (qs->Qs[0].binaryLength[i] > maxlen)
		{
			maxlen = qs->Qs[0].binaryLength[i];
		}
	}
	cout << *tolbits << endl;
	buildColPattern(*tolbits, qs->Qs[0].binaryLength, qs->Qs[0].columnNumber);
	for (int i = 0; i < 20; i++)
	{
		MadeBuffer[i] = new MADENet;
		deepCopyMade(C->Mnet, MadeBuffer[i]);
	}
	// cout<<"deepCPDONE"<<endl;
	// exit(1);
	int TP = 0, FP = 0, TN = 0, FN = 0; // tp:��ȷʶ��small

	for (int i = 0; i < up; i++)
	{
		// i =1;
		// cout<<"QID:"<<i<<endl;
		Query qi = qs->Qs[i];
		for (int j = 0; j < 20; j++)
		{
			QueryBuffer[j] = new Query;
			deepCopyQuery(&qi, QueryBuffer[j]);
		}
		int realcard, estcard;
		realcard = qid2TrueNumber[i];
		bool *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
		ZTuple *ZT0 = makeZT(&zencode0[1], *tolbits);
		bool *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
		ZTuple *ZT1 = makeZT(&zencode1[1], *tolbits);
		auto queryfirst = Clock::now(); // ��ʱ��ʼ
		auto t3 = Clock::now();			// ��ʱ��ʼ
		for (int j = 0; j < *tolbits; j++)
		{
			GLBZEnc0[j] = zencode0[j + 1];
			GLBZEnc1[j] = zencode1[j + 1];
		}
		int probN = 25;
		if (ProbeTest != -1)
		{
			probN = ProbeTest;
		}
		float cdfapx = probeParrelN(&zencode0[1], &zencode1[1], *tolbits, C, qi, probN);
		// probeParrelN(&zencode0[1], &zencode1[1], *tolbits, C, qi, probN);
		// cout << (realcard + 0.0) / C->Mnet->zdr << endl;
		auto t3Mid = Clock::now(); // ��ʱ��ʼ

		// cout <<"QID:"<<i << "Approx: " << cdfapx << "Tks: "<< (std::chrono::duration_cast<std::chrono::nanoseconds>(t3Mid - t3).count())<<endl;
		// exit(1);
		CDFEstTime += (std::chrono::duration_cast<std::chrono::nanoseconds>(t3Mid - t3).count());

		double bound = bEST;
		if (cdfapx < bound)
		{
			if ((realcard + 0.0) / C->Mnet->zdr < bound)
			{
				TP += 1;
			}
			else
			{
				FP += 1;
			}
		}
		else
		{
			if ((realcard + 0.0) / C->Mnet->zdr < bound)
			{
				FN += 1;
			}
			else
			{
				TN += 1;
			}
		}
		if (ProbeTest != -1)
		{
			continue;
		}
		if (cdfapx < bEST)
		{
			estcard = paraRangeQ(C, qi, zencode0, zencode1, ZT0, ZT1);
			estcard = realcard;
		}
		else
		{
			estcard = cardEstimate(C->Mnet, qi, sampleN);
		}
		auto t3d = Clock::now(); // ��ʱend
		long delta = (std::chrono::duration_cast<std::chrono::nanoseconds>(t3d - t3).count());
		timesum += delta;
		// continue;
		// estcard = cdfapx * C->Mnet->zdr;
		//  cout << "Card Est time:" << delta << endl;
		//  cout << estcard << endl;
		// timesum += (std::chrono::duration_cast<std::chrono::nanoseconds>(t3d - t3).count());
		//
		auto queryend = Clock::now(); // ��ʱ��ʼ
		// cout << "Qid" << qi.queryid << "\tRealCard:" << realcard << "\tEstCard:" << estcard << " P :" << pErrorcalculate(estcard, realcard) << endl;
		// ofs << "Qid" << qi.queryid << "\tRealCard:" << realcard << "\tEstCard:" << estcard << " P :" << pErrorcalculate(estcard, realcard) << endl;
		queryestTime += (std::chrono::duration_cast<std::chrono::nanoseconds>(queryend - queryfirst).count());
		// cout << "Query est time:" << to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(queryend - queryfirst).count()) << "ns" << endl;
		// ofs << "Qid" << qi.queryid << "\tRealCard:" << realcard << "\tEstCard:" << estcard << " P :" << pErrorcalculate(estcard, realcard) << endl;
		pdist.push_back(pErrorcalculate(estcard, realcard));
		// exit(1);
	}
	cout << "CDF ProbeTime:" << CDFEstTime / qs->queryNumber << endl;
	ofs << "CDF ProbeTime:" << CDFEstTime / qs->queryNumber << endl;
	cout << "TP TN FP FN" << endl;
	ofs << "TP TN FP FN" << endl;

	cout << TP << " " << TN << " " << FP << " " << FN << "Acc: " << (TP + TN + 0.0) / qs->queryNumber << endl;
	ofs << TP << " " << TN << " " << FP << " " << FN << "	Acc: " << (TP + TN + 0.0) / qs->queryNumber << endl;

	if (ProbeTest != -1)
	{
		return;
	}
	cout << "AVG APROX T" << timesum / qs->queryNumber << endl;
	ofs << "AVG APROX T" << timesum / qs->queryNumber << endl;

	// return;
	// exit(1);
	cout << CDFEstTime / qs->queryNumber << endl;
	sort(pdist.begin(), pdist.end());
	p50 = pdist[((int)(0.5 * pdist.size()))];
	p95 = pdist[((int)(0.95 * pdist.size()))];
	p90 = pdist[((int)(0.90 * pdist.size()))];
	p99 = pdist[((int)(0.99 * pdist.size()))];
	// p99 = pdist[((int)(0.99 * pdist.size()))];
	float pmax = pdist[pdist.size() - 1];
	cout << "P50\tP90\tP95\tP99\tPmax\tAvgT" << endl;
	ofs << "P50\tP90\tP95\tP99\tPmax\tAvgT" << endl;
	ofs << p50 << " " << p90 << ", " << p95 << ", " << p99 << ", " << pmax << ", " << endl;
	cout << p50 << "\t" << p90 << "\t" << p95 << '\t' << p99 << '\t' << pmax << '\t' << queryestTime / qs->queryNumber << endl;
}

void testBTBulkInsert()
{
	BPlusTree root = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	for (int i = 0; i < 19; i++)
	{
		string filePathi = "./data/ZD" + to_string(i) + ".txt";
		cout << "------------------------\n";
		ZTab *ZT = loadZD(filePathi);
		cout << filePathi << " Loaded" << endl;
		auto t3 = Clock::now(); // ��ʱ��ʼ

		root = bulkInsert(root, ZT);
		cout << "Ins done" << endl;
		auto thalf = Clock::now();
		cout << "Static Ins" << (std::chrono::duration_cast<std::chrono::milliseconds>(thalf - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		cout << "checking correctness" << endl;
		BPlusNode *curptr = root;
		while (curptr->Children[0] != NULL)
		{
			curptr = curptr->Children[0];
		}
		int aggre = 0;
		while (curptr != NULL)
		{
			aggre += curptr->KeyNum;
			curptr = curptr->Next;
		}
		cout << "Toltal elements: " << aggre << endl;
	}
}
void testCardInsert()
{
	// BPlusTree root = NULL;
	BPlusNode *LinkEDList = NULL;
	ofs << "MergeTime; MaintainTime" << endl;
	typedef std::chrono::high_resolution_clock Clock;
	for (int i = 0; i <= 10; i++)
	{
		string filePathi = "./data/ZD" + to_string(i) + ".txt";
		cout << "------------------------\n";
		ZTab *ZT = loadZD(filePathi);
		cout << filePathi << " Loaded" << endl;
		MADENet *Net = NULL;
		cout << "NOTFILE" << endl;
		exit(1);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZT);
		auto thalf = Clock::now();
		cout << "Merge done" << endl;
		cout << "AllTime: " << (std::chrono::duration_cast<std::chrono::milliseconds>(thalf - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		ofs << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << "; ";
		t3 = Clock::now(); // ��ʱ��ʼ
		CardIndex *CI = LinkedList2CardIndex(LinkEDList, Net);
		thalf = Clock::now();
		cout << "AllTime: " << (std::chrono::duration_cast<std::chrono::milliseconds>(thalf - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		ofs << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		cout << "checking correctness" << endl;
		BPlusNode *curptr = CI->Head;
		string queryFilePath = "./query/PQ" + to_string(i) + ".txt";
		cout << queryFilePath << endl;
		// testCardIndexPointQuery(CI, queryFilePath);
	}
}

void testIndexQueryPerformance()
{
	BPlusNode *LinkEDList = NULL;
	ofs << "MergeTime; MaintainTime" << endl;
	typedef std::chrono::high_resolution_clock Clock;
	for (int i = 0; i <= 15; i++)
	{
		string filePathi = "./data/ZD" + to_string(i) + ".txt";
		cout << "------------------------\n";
		ZTab *ZT = loadZD(filePathi);
		cout << filePathi << " Loaded" << endl;
		MADENet *Net = NULL;
		cout << "NOTFILE" << endl;
		exit(1);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZT);
		auto thalf = Clock::now();
		cout << "Merge done" << endl;
		cout << "AllTime: " << (std::chrono::duration_cast<std::chrono::milliseconds>(thalf - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		ofs << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << "; ";
		t3 = Clock::now(); // ��ʱ��ʼ
		CardIndex *CI = LinkedList2CardIndex(LinkEDList, Net);
		thalf = Clock::now();
		cout << "AllTime: " << (std::chrono::duration_cast<std::chrono::milliseconds>(thalf - t3).count()) << " In average(ns/t):" << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		ofs << (std::chrono::duration_cast<std::chrono::nanoseconds>(thalf - t3).count()) / (ZT->r + 0.0) << endl;
		cout << "checking correctness via rqnge query" << endl;
		BPlusNode *curptr = CI->Head;
		string queryFilePath = "./query/Q" + to_string(i) + ".txt";
		cout << queryFilePath << endl;
		// testCardIndexPointQuery(CI, queryFilePath);
		testCardIndexRangeQuery(CI, queryFilePath);
	}
}
void SampleFromLinkedList(int NeedNum, int TolNum, BPlusNode *LinkEDList, vector<ZTuple *> *SampledResult)
{
	cout << "Sample start:" << LinkEDList << " " << TolNum << endl;
	if (LinkEDList == NULL || TolNum == 0 || NeedNum == 0)
	{
		return;
	}
	float sampleRatio = (NeedNum + 0.0) / TolNum;

	int cnt = 0;
	for (BPlusNode *ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		for (int i = 0; i < ptr->KeyNum; i++)
		{
			int samFlag = randG(sampleRatio);
			if (samFlag)
			{
				(*SampledResult).push_back(ptr->Key[i]);
				cnt++;
				if (cnt > NeedNum)
				{
					return;
				}
			}
		}
	}
}
void testInsertedCardPerformance()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string postfix = ".txt";
	for (int i = 0; i <= 9; i++)
	{
		string filePathi = "./ZD";
		filePathi += to_string(i);
		filePathi += postfix;
		cout << "------------------------\n";
		ZTab *ZT = loadZD(filePathi);
		long long InsertTime = 0;
		auto tMegStart = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZT);
		auto tMegEnd = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
		cout << "Merge Takes(ns)" << InsertTime << endl;
		vector<ZTuple *> sam;
		SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
		curLen += ZT->r;
		cout << "MergeDone" << endl;
		cout << filePathi << " Loaded" << endl;
		float *arr = new float[ZT->r * ZT->c];
		float *arrI = new float[sam.size() * ZT->c];
		cout << "OK2d" << endl;
		for (int i = 0; i < ZT->r; i++)
		{
			for (int j = 0; j < ZT->c; j++)
			{
				arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
			}
		}
		for (int i = 0; i < sam.size(); i++)
		{
			for (int j = 0; j < ZT->c; j++)
			{
				arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
			}
		}
		cout << "arr inited" << endl;
		torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
		torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]

		cout << "ten inited" << endl;
		auto tPre = Clock::now(); // ��ʱ��ʼ
		if (Net == NULL)
		{
			Net = TrainOnD(&D_Ten, 30, 20);
		}
		else
		{
			Net = IncrementalTraining(&D_Ten, &I_Ten, Net);
		}
		auto t3 = Clock::now(); // ��ʱ��ʼ
		std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
		CardIndex *CI;
		string queryFilePath = "./Q" + to_string(i) + ".txt";
		auto TLink = Clock::now(); // ��ʱ��ʼ
		CI = LinkedList2CardIndex(LinkEDList, Net);
		auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
		cout << CI->Mnet->zdr << endl;
		std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		cout << queryFilePath << endl;
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		cout << "Intotal: " << InsertTime << endl;
		testCardPerformance(CI, queryFilePath);
		exit(1);
		// testCardIndexRangeQuery(CI, queryFilePath);
	}
}
void testFullCardPerformanceDMV()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]

	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 5);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);

	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	/*
	cout << queryFilePath << endl;
	ofs << queryFilePath << endl;
	testCardPerformance(CI, queryFilePath);
	*/
	string queryFilePath = "./query/DMV1";
	cout << queryFilePath << endl;

	testCardPerformance(CI, queryFilePath);
	// cout << queryFilePath << endl;
	////testCardIndexRangeQuery(CI, queryFilePath);
	queryFilePath = "./query/DMV001";
	cout << queryFilePath << endl;

	// cout << queryFilePath << endl;
	testCardPerformance(CI, queryFilePath);
	// testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/DMV00001";
	cout << queryFilePath << endl;

	testCardPerformance(CI, queryFilePath);
	////	ofs<< queryFilePath << endl;
	////	//testCardIndexRangeQuery(CI, queryFilePath);
	// testCardPerformance(CI, queryFilePath);
	////exit(1);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testFullRangeQPerformance()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/power";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]

	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 20);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	// int idx = 0;
	// int wrongN = 0;
	// int less = 0;
	// long long timeSum = 0;
	// for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	// {
	// 	idx++;
	// 	auto tPStart = Clock::now(); // ��ʱ��ʼ
	// 	auto page = CardIndexLeafGet(CI, ptr->Key[0]);
	// 	auto tPEnd = Clock::now(); // ��ʱ��ʼ
	// 	timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	// }
	// cout << timeSum << endl;
	// cout << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
	// ofs<< "Avg PointQueryTime(ns):" << timeSum / idx << endl;

	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	string queryFilePath = "./query/power1";
	// testCardIndexRangeQuery(CI, queryFilePath);
	// // cout << queryFilePath << endl;
	// // ofs << queryFilePath << endl;
	// // testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/power001";
	cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardIndexRangeQuery(CI, queryFilePath);
	queryFilePath = "./query/power00001";
	cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardIndexRangeQuery(CI, queryFilePath);

	// testCardPerformance(CI, queryFilePath);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testFullRangeQPerformanceDMV()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]

	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 2);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
	ofs << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
	string queryFilePath = "./query/DMV1";
	// testCardIndexRangeQuery(CI, queryFilePath);
	// cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	// testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/DMV001";
	cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardIndexRangeQuery(CI, queryFilePath);
	queryFilePath = "./query/DMV00001";
	cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardIndexRangeQuery(CI, queryFilePath);
	// testCardPerformance(CI, queryFilePath);
	// testCardIndexRangeQuery(CI, queryFilePath);
}

void testFullCardPerformanceOSM()
{

	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/OSM";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	// LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	// SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	cout << ZT->r << " " << ZT->c << endl;
	int rhalf = ZT->r / 2;
	int lhalf = ZT->r - rhalf;
	cout << "MALLOCing sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	float *arr1 = new float[rhalf * ZT->c];
	float *arr2 = new float[lhalf * ZT->c];
	cout << "MALLOCED sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	cout << "OK2d" << endl;
	for (int i = 0; i < rhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr1[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < lhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr2[i * ZT->c + j] = ZT->D[i + rhalf]->bin[j] + 0.0;
		}
	}
	torch::Tensor D_Ten1 = torch::from_blob(arr1, {rhalf, ZT->c}); // shape��[r, c]
	torch::Tensor D_Ten2 = torch::from_blob(arr2, {lhalf, ZT->c}); // shape��[r, c]
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = IncrementalTraining(&D_Ten1, &D_Ten2, Net);
	delete arr1;
	delete arr2;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	string queryFilePath = "./query/osm1";
	// cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/osm001";
	cout << queryFilePath << endl;

	// ofs << queryFilePath << endl;
	// testCardIndexRangeQuery(CI, queryFilePath);
	testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/osm00001";
	//	ofs<< queryFilePath << endl;
	//	//testCardIndexRangeQuery(CI, queryFilePath);
	testCardPerformance(CI, queryFilePath);
	// exit(1);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testFullRangeQPerformanceOSM()
{

	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/OSM";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	// LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	// SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	cout << ZT->r << " " << ZT->c << endl;
	int rhalf = ZT->r / 2;
	int lhalf = ZT->r - rhalf;
	cout << "MALLOCing sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	float *arr1 = new float[rhalf * ZT->c];
	float *arr2 = new float[lhalf * ZT->c];
	cout << "MALLOCED sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	cout << "OK2d" << endl;
	for (int i = 0; i < rhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr1[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < lhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr2[i * ZT->c + j] = ZT->D[i + rhalf]->bin[j] + 0.0;
		}
	}
	torch::Tensor D_Ten1 = torch::from_blob(arr1, {rhalf, ZT->c}); // shape��[r, c]
	torch::Tensor D_Ten2 = torch::from_blob(arr2, {lhalf, ZT->c}); // shape��[r, c]
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = IncrementalTraining(&D_Ten1, &D_Ten2, Net);
	delete arr1;
	delete arr2;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
	ofs << "Avg PointQueryTime(ns):" << timeSum / idx << endl;

	string queryFilePath = "./query/osm1";
	// testCardIndexRangeQuery(CI, queryFilePath);
	queryFilePath = "./query/osm001";
	cout << queryFilePath << endl;
	testCardIndexRangeQuery(CI, queryFilePath);
	queryFilePath = "./query/osm00001";
	testCardIndexRangeQuery(CI, queryFilePath);
	// exit(1);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testProbePerfoemanceOSM()
{

	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/OSM";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	// LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	// SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	cout << ZT->r << " " << ZT->c << endl;
	int rhalf = ZT->r / 2;
	int lhalf = ZT->r - rhalf;
	cout << "MALLOCing sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	float *arr1 = new float[rhalf * ZT->c];
	float *arr2 = new float[lhalf * ZT->c];
	cout << "MALLOCED sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	cout << "OK2d" << endl;
	for (int i = 0; i < rhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr1[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < lhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr2[i * ZT->c + j] = ZT->D[i + rhalf]->bin[j] + 0.0;
		}
	}
	torch::Tensor D_Ten1 = torch::from_blob(arr1, {rhalf, ZT->c}); // shape��[r, c]
	torch::Tensor D_Ten2 = torch::from_blob(arr2, {lhalf, ZT->c}); // shape��[r, c]
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = IncrementalTraining(&D_Ten1, &D_Ten2, Net);
	delete arr1;
	delete arr2;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	string queryFilePath = "./query/osm1";
	for (int pD = 3; pD <= 24; pD += 3)
	{
		cout << "\n==================================\n";
		cout << "ProbDep: " << pD << endl;
		queryFilePath = "./query/osm1";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/osm001";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/osm00001";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		cout << "\n==================================\n";
	}
	// exit(1);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testPointQPerformanceOSM()
{

	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/OSM";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	// LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	// SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	cout << ZT->r << " " << ZT->c << endl;
	int rhalf = ZT->r / 2;
	int lhalf = ZT->r - rhalf;
	cout << "MALLOCing sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	float *arr1 = new float[rhalf * ZT->c];
	float *arr2 = new float[lhalf * ZT->c];
	cout << "MALLOCED sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	cout << "OK2d" << endl;
	for (int i = 0; i < rhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr1[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < lhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr2[i * ZT->c + j] = ZT->D[i + rhalf]->bin[j] + 0.0;
		}
	}
	torch::Tensor D_Ten1 = torch::from_blob(arr1, {rhalf, ZT->c}); // shape��[r, c]
	torch::Tensor D_Ten2 = torch::from_blob(arr2, {lhalf, ZT->c}); // shape��[r, c]
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = IncrementalTraining(&D_Ten1, &D_Ten2, Net);
	delete arr1;
	delete arr2;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << idx << endl;
	cout << "Avg PQTime:" << timeSum / idx << endl;
}
void testProbePerfoemance()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/power";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(0, ZT->r, LinkEDList, &sam);
	curLen += ZT->r;
	cout << ZT->r << " " << sam.size() << endl;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]
	cout << D_Ten.sizes()[0] << " " << I_Ten.sizes()[0] << endl;
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 10);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "TreeSize:" << mallocedSize / (1e6) << endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	string queryFilePath = "./query/power1";
	for (int pD = 0; pD <= 25; pD += 5)
	{
		cout << "\n==================================\n";
		ofs << "\n==================================\n";
		cout << "ProbDep: " << pD << endl;
		ofs << "ProbDep: " << pD << endl;
		queryFilePath = "./query/power1";
		cout << queryFilePath << endl;
		ofs << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/power001";
		ofs << queryFilePath << endl;
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/power00001";
		cout << queryFilePath << endl;
		ofs << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		cout << "\n==================================\n";
	}
}
void testProbePerfoemanceDMV()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	// vector<ZTuple* > sam;
	// SampleFromLinkedList(0, ZT->r, LinkEDList, &sam);
	curLen += ZT->r;
	// cout << ZT->r << " " << sam.size() << endl;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	// float* arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	// for (int i = 0; i < sam.size(); i++) {
	//	for (int j = 0; j < ZT->c; j++) {
	//		arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
	//	}
	// }
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c}); // shape��[r, c]
	// torch::Tensor I_Ten = torch::from_blob(arrI, { (long long)sam.size()  ,ZT->c }); // shape��[r, c]
	// cout << D_Ten.sizes()[0] << " " << I_Ten.sizes()[0] << endl;
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 10);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	string queryFilePath = "./query/DMV1";
	for (int pD = 0; pD <= 25; pD += 5)
	{
		cout << "\n==================================\n";
		cout << "ProbDep: " << pD << endl;
		queryFilePath = "./query/DMV1";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/DMV001";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		queryFilePath = "./query/DMV00001";
		cout << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath, pD);
		cout << "\n==================================\n";
	}
}
void testFullCardPerformance()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/power";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(0, ZT->r, LinkEDList, &sam);
	curLen += ZT->r;
	cout << ZT->r << " " << sam.size() << endl;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]
	cout << D_Ten.sizes()[0] << " " << I_Ten.sizes()[0] << endl;
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 10);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	string queryFilePath = "./query/power1";
	cout << queryFilePath << endl;
	// ofs << queryFilePath << endl;
	testCardPerformance(CI, queryFilePath);
	queryFilePath = "./query/power001";
	cout << queryFilePath << endl;

	// ofs << queryFilePath << endl;
	// testCardIndexRangeQuery(CI, queryFilePath);
	testCardPerformance(CI, queryFilePath);

	queryFilePath = "./query/power00001";
	cout << queryFilePath << endl;

	//	ofs<< queryFilePath << endl;
	//	//testCardIndexRangeQuery(CI, queryFilePath);
	testCardPerformance(CI, queryFilePath);
	// exit(1);
	// testCardIndexRangeQuery(CI, queryFilePath);
}
void testBIGMIN()
{
	maxlen = 4;
	int colNum = 2;
	int binL[2] = {4, 3};
	*tolbits = 7;
	buildColPattern(*tolbits, binL, colNum);
	for (int i = 0; i < 7; i++)
	{
		cout << colPattern[i] << " ";
	}
	cout << endl;
	for (auto v : Zidx2Col)
	{
		cout << v << endl;
	}
	bool minz[] = {0, 0, 1, 1, 0, 1, 1};
	bool maxz[] = {1, 1, 0, 0, 1, 1, 0};
	for (int v = 56; v <= 73; v++)
	{
		cout << "Test value:" << v << endl;
		bool zv[200];
		int vc = v;
		for (int i = 0; i < 7; i++)
		{
			zv[6 - i] = vc % 2;
			vc /= 2;
		}
		int bitl = 7;
		int binL[] = {4, 3};
		bool *bgm = getBIGMINF(minz, maxz, zv, bitl, binL, 2);
		bool *ltm = getLITMAXF(minz, maxz, zv, bitl, binL, 2);
		int sum = 0;
		for (int i = 0; i < bitl; i++)
		{
			cout << bgm[i];
			sum *= 2;
			sum += bgm[i];
		}
		cout << endl;
		cout << "BIGMIN Value:" << sum << endl;
		sum = 0;
		for (int i = 0; i < bitl; i++)
		{
			cout << ltm[i];
			sum *= 2;
			sum += ltm[i];
		}
		cout << endl;
		cout << "LITMAX Value:" << sum << endl;
		sum = 0;
	}
}
void check(BPlusNode *subN)
{
	if (subN == NULL)
	{
		return;
	}
	else
	{
		if (subN->Zmin != subN->Key[0]->z32)
		{
			cout << "find BUG" << endl;
			exit(1);
		}
		else
		{
			for (int i = 0; i < subN->KeyNum; i++)
			{
				check(subN->Children[i]);
			}
		}
	}
}
void testPointPageAccesss()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]

	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 2);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	cout << "checking CI" << endl;

	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << idx << endl;
	cout << "Avg PQTime:" << timeSum / idx << endl;
}
void testUpdateCardPerformance()
{
	typedef std::chrono::high_resolution_clock Clock;
	string filePathi = "./data/power";
	for (int i = 5; i <= 20; i *= 2)
	{
		cout << "\n\n=======================================\n"
			 << endl;
		ofs << "\n\n=======================================\n"
			<< endl;
		int curLen = 0;
		long long InsertTime = 0;
		BPlusNode *LinkEDList = NULL;
		MADENet *Net = NULL;
		string filePathOld = filePathi;
		string filePathNew = "./data/Zpower" + to_string(i);
		cout << filePathOld << " " << filePathNew << endl;
		ZTab *ZTOld = loadZDBin(filePathOld);
		ZTab *ZTNew = loadZD(filePathNew);
		cout << "Both File Loaded" << endl;
		cout << "First, we train on old Data" << endl;
		ofs << "Train On Old D" << endl;
		int r = ZTOld->r;
		curLen += ZTNew->r;
		float *arr1 = new float[r * ZTOld->c];
		for (int i = 0; i < r; i++)
		{
			for (int j = 0; j < ZTOld->c; j++)
			{
				arr1[i * ZTOld->c + j] = ZTOld->D[i]->bin[j] + 0.0;
			}
		}
		torch::Tensor D_Old = torch::from_blob(arr1, {r, ZTOld->c}); // shape��[r, c]
		cout << "Tensor inited" << endl;
		auto tPre = Clock::now(); // ��ʱ��ʼ
		Net = TrainOnD(&D_Old, 30, 20);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		std::cout << "\nOld-TrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
		CardIndex *CI;
		auto TLink = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZTOld);
		CI = LinkedList2CardIndex(LinkEDList, Net);
		auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		cout << "Intotal: " << InsertTime << endl;
		ofs << "Intotal: " << InsertTime << endl;
		cout << "Now for incremental" << endl;
		InsertTime = 0;
		cout << "Sample from the Old" << endl;
		vector<ZTuple *> sam;
		SampleFromLinkedList(ZTNew->r, ZTOld->r, LinkEDList, &sam);
		curLen += ZTNew->r;
		cout << "MergeDone" << endl;
		cout << filePathi << " Loaded" << endl;
		float *arr = new float[ZTNew->r * ZTNew->c];
		float *arrO = new float[sam.size() * ZTNew->c];
		cout << "OK2d" << endl;
		for (int i = 0; i < ZTNew->r; i++)
		{
			for (int j = 0; j < ZTNew->c; j++)
			{
				arr[i * ZTNew->c + j] = ZTNew->D[i]->bin[j] + 0.0;
			}
		}
		for (int i = 0; i < sam.size(); i++)
		{
			for (int j = 0; j < ZTNew->c; j++)
			{
				arrO[i * ZTNew->c + j] = sam[i]->bin[j] + 0.0;
			}
		}
		torch::Tensor Ins_Ten = torch::from_blob(arr, {ZTNew->r, ZTNew->c});			   // shape��[r, c]
		torch::Tensor Old_Ten = torch::from_blob(arrO, {(long long)sam.size(), ZTNew->c}); // shape��[r, c]
		tPre = Clock::now();															   // ��ʱ��ʼ
		Net = IncrementalTraining(&Old_Ten, &Ins_Ten, Net, ZTOld->r, ZTNew->r);
		auto tPost = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count();
		cout << "Incremental Training Net Tks:" << std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count() << endl;
		ofs << "Incremental Training Net Takes:" << std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count() << endl;

		TLink = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZTNew);
		CI = LinkedList2CardIndex(LinkEDList, Net);
		TLinkEnd = Clock::now(); // ��ʱ��ʼ
		cout << CI->Mnet->zdr << endl;
		std::cout << "\nMerge and Link Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		ofs << "\nMerge and Link Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		cout << "Incremental Train Takses(ns): " << InsertTime << endl;
		cout << "Incremental Train Takses(s): " << InsertTime / 1e9 << endl;
		ofs << "Incremental Train Takses(ns): " << InsertTime << endl;
		ofs << "Incremental Train Takses(s): " << InsertTime / 1e9 << endl;
		string queryFilePath = "./query/power_" + to_string(i) + "-CI";
		cout << queryFilePath << endl;
		// ofs << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath);
		// cout<<"Testing PQ"<<endl;
		// testCorrectPointQuery(LinkEDList,CI);
	}
	exit(1);
}
void testUpdateCardPerformanceDMV()
{
	typedef std::chrono::high_resolution_clock Clock;
	string filePathi = "./data/DMV";
	for (int i = 5; i <= 20; i *= 2)
	{
		cout << "\n\n=======================================\n"
			 << endl;
		ofs << "\n\n=======================================\n"
			<< endl;
		int curLen = 0;
		long long InsertTime = 0;
		BPlusNode *LinkEDList = NULL;
		MADENet *Net = NULL;
		string filePathOld = filePathi;
		string filePathNew = "./data/ZDMV" + to_string(i);
		cout << filePathOld << " " << filePathNew << endl;
		ZTab *ZTOld = loadZDBin(filePathOld);
		ZTab *ZTNew = loadZD(filePathNew);
		cout << "First, we train on old Data" << endl;
		ofs << "Train On Old D" << endl;
		int r = ZTOld->r;
		curLen += ZTNew->r;

		float *arr1 = new float[r * ZTOld->c];
		for (int i = 0; i < r; i++)
		{
			for (int j = 0; j < ZTOld->c; j++)
			{
				arr1[i * ZTOld->c + j] = ZTOld->D[i]->bin[j] + 0.0;
			}
		}
		torch::Tensor D_Old = torch::from_blob(arr1, {r, ZTOld->c}); // shape��[r, c]
		cout << "Tensor inited" << endl;
		auto tPre = Clock::now(); // ��ʱ��ʼ
		Net = TrainOnD(&D_Old, 30, 20);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		std::cout << "\nOld-TrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
		CardIndex *CI;
		auto TLink = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZTOld);
		CI = LinkedList2CardIndex(LinkEDList, Net);
		auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		cout << "Intotal: " << InsertTime << endl;
		ofs << "Intotal: " << InsertTime << endl;
		cout << "Now for incremental" << endl;
		InsertTime = 0;
		cout << "Sample from the Old" << endl;
		vector<ZTuple *> sam;
		SampleFromLinkedList(ZTNew->r, ZTOld->r, LinkEDList, &sam);
		curLen += ZTNew->r;
		cout << "MergeDone" << endl;
		cout << filePathi << " Loaded" << endl;
		float *arr = new float[ZTNew->r * ZTNew->c];
		float *arrO = new float[sam.size() * ZTNew->c];
		cout << "OK2d" << endl;
		for (int i = 0; i < ZTNew->r; i++)
		{
			for (int j = 0; j < ZTNew->c; j++)
			{
				arr[i * ZTNew->c + j] = ZTNew->D[i]->bin[j] + 0.0;
			}
		}
		for (int i = 0; i < sam.size(); i++)
		{
			for (int j = 0; j < ZTNew->c; j++)
			{
				arrO[i * ZTNew->c + j] = sam[i]->bin[j] + 0.0;
			}
		}
		torch::Tensor Ins_Ten = torch::from_blob(arr, {ZTNew->r, ZTNew->c});			   // shape��[r, c]
		torch::Tensor Old_Ten = torch::from_blob(arrO, {(long long)sam.size(), ZTNew->c}); // shape��[r, c]
		tPre = Clock::now();															   // ��ʱ��ʼ
		Net = IncrementalTraining(&Old_Ten, &Ins_Ten, Net, ZTOld->r, ZTNew->r);
		auto tPost = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count();
		cout << "Incremental Training Net Tks:" << std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count() << endl;
		ofs << "Incremental Training Net Takes:" << std::chrono::duration_cast<std::chrono::nanoseconds>(tPost - tPre).count() << endl;

		TLink = Clock::now(); // ��ʱ��ʼ
		LinkEDList = MergeLinkedList(LinkEDList, ZTNew);
		CI = LinkedList2CardIndex(LinkEDList, Net);
		TLinkEnd = Clock::now(); // ��ʱ��ʼ
		cout << CI->Mnet->zdr << endl;
		std::cout << "\nMerge and Link Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		ofs << "\nMerge and Link Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		cout << "Incremental Train Takses(ns): " << InsertTime << endl;
		cout << "Incremental Train Takses(s): " << InsertTime / 1e9 << endl;
		ofs << "Incremental Train Takses(ns): " << InsertTime << endl;
		ofs << "Incremental Train Takses(s): " << InsertTime / 1e9 << endl;
		string queryFilePath = "./query/DMV_" + to_string(i) + "-CI";
		cout << queryFilePath << endl;
		ofs << queryFilePath << endl;
		testCardPerformance(CI, queryFilePath);
	}
	exit(1);
}

void testScale(int num)
{
	ofs << "Scale On Num:" << num << endl;
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/OSM";
	ZTab *ZT = loadZDBin(filePathi, 1, num);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	// LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	// SampleFromLinkedList(ZT->r, curLen, LinkEDList, &sam);
	curLen += ZT->r;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	cout << ZT->r << " " << ZT->c << endl;
	int rhalf = ZT->r / 2;
	int lhalf = ZT->r - rhalf;
	cout << "MALLOCing sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	float *arr1 = new float[rhalf * ZT->c];
	float *arr2 = new float[lhalf * ZT->c];
	cout << "MALLOCED sze:" << (rhalf * ZT->c * 4) / 1e9 << " " << (lhalf * ZT->c * 4) / 1e9 << endl;
	cout << "OK2d" << endl;
	for (int i = 0; i < rhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr1[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < lhalf; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr2[i * ZT->c + j] = ZT->D[i + rhalf]->bin[j] + 0.0;
		}
	}
	torch::Tensor D_Ten1 = torch::from_blob(arr1, {rhalf, ZT->c}); // shape��[r, c]
	torch::Tensor D_Ten2 = torch::from_blob(arr2, {lhalf, ZT->c}); // shape��[r, c]
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = IncrementalTraining(&D_Ten1, &D_Ten2, Net);
	delete arr1;
	delete arr2;
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(s): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() / (1e9) << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	cout << "MalS:" << mallocedSize << endl;
	mallocedSize = 0;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime / (1e9) << endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	int idx = 0;
	int wrongN = 0;
	int less = 0;
	long long timeSum = 0;
	for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
	{
		idx++;
		auto tPStart = Clock::now(); // ��ʱ��ʼ
		auto page = CardIndexLeafGet(CI, ptr->Key[0]);
		auto tPEnd = Clock::now(); // ��ʱ��ʼ
		timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
	}
	cout << timeSum << endl;
	cout << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
	ofs << "Avg PointQueryTime(ns):" << timeSum / idx << endl;
}
void adaptiveStudy(){
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(0, ZT->r, LinkEDList, &sam);
	curLen += ZT->r;
	cout << ZT->r << " " << sam.size() << endl;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]
	cout << D_Ten.sizes()[0] << " " << I_Ten.sizes()[0] << endl;
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ
	Net = TrainOnD(&D_Ten, 30, 10);
	auto t3 = Clock::now(); // ��ʱ��ʼ
	std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();
	CardIndex *CI;
	auto TLink = Clock::now(); // ��ʱ��ʼ
	CI = LinkedList2CardIndex(LinkEDList, Net);
	auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
	std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
	cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
	cout << "Intotal: " << InsertTime << endl;
	ofs << "Intotal: " << InsertTime << endl;
	for (float best = 0.001; best<=1;best*=100)
	{
		bEST = best;
		cout<<"bEST:"<<best<<endl;
		string firstStr = "./query/DMV";
		for(int j = 0;j<=6;j++){
			string queryFilePath = firstStr+"1";
			firstStr+="0";
			cout << queryFilePath << endl;
			testCardPerformance(CI, queryFilePath);
		}
		// string queryFilePath = "./query/DMV001";
		// cout << queryFilePath << endl;
		// testCardPerformance(CI, queryFilePath);
	}
	exit(1);
}
void varaStudy()
{
	BPlusNode *LinkEDList = NULL;
	typedef std::chrono::high_resolution_clock Clock;
	MADENet *Net = NULL;
	int curLen = 0;
	string filePathi = "./data/DMV";
	ZTab *ZT = loadZDBin(filePathi);
	cout << "loaded" << endl;
	long long InsertTime = 0;
	auto tMegStart = Clock::now(); // ��ʱ��ʼ
	LinkEDList = MergeLinkedList(LinkEDList, ZT);
	auto tMegEnd = Clock::now(); // ��ʱ��ʼ
	InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tMegEnd - tMegStart).count();
	cout << "Merge Takes(ns)" << InsertTime << endl;
	vector<ZTuple *> sam;
	SampleFromLinkedList(0, ZT->r, LinkEDList, &sam);
	curLen += ZT->r;
	cout << ZT->r << " " << sam.size() << endl;
	cout << "MergeDone" << endl; // 12300116
	cout << filePathi << " Loaded" << endl;
	float *arr = new float[ZT->r * ZT->c];
	float *arrI = new float[sam.size() * ZT->c];
	cout << "OK2d" << endl;
	for (int i = 0; i < ZT->r; i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arr[i * ZT->c + j] = ZT->D[i]->bin[j] + 0.0;
		}
	}
	for (int i = 0; i < sam.size(); i++)
	{
		for (int j = 0; j < ZT->c; j++)
		{
			arrI[i * ZT->c + j] = sam[i]->bin[j] + 0.0;
		}
	}
	cout << "arr inited" << endl;
	torch::Tensor D_Ten = torch::from_blob(arr, {ZT->r, ZT->c});				  // shape��[r, c]
	torch::Tensor I_Ten = torch::from_blob(arrI, {(long long)sam.size(), ZT->c}); // shape��[r, c]
	cout << D_Ten.sizes()[0] << " " << I_Ten.sizes()[0] << endl;
	cout << "Tensor inited" << endl;
	auto tPre = Clock::now(); // ��ʱ��ʼ

	for (int linkedN = 1; linkedN <= 32; linkedN *= 2)
	{
		cout<<"LinkedN: "<<linkedN<<endl;
		Net = TrainOnD(&D_Ten, linkedN, 10);
		auto t3 = Clock::now(); // ��ʱ��ʼ
		std::cout << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
		ofs << "\nTrainingNet Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count() << std::endl;
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - tPre).count();

		CardIndex *CI;
		auto TLink = Clock::now(); // ��ʱ��ʼ
		CI = LinkedList2CardIndex(LinkEDList, Net);
		auto TLinkEnd = Clock::now(); // ��ʱ��ʼ
		InsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count();
		std::cout << "\nLink Takes(ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(TLinkEnd - TLink).count() << std::endl;
		cout << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
		ofs << "Tree size in total(MB):" << mallocedSize / (1e6) << endl;
		cout << "Intotal: " << InsertTime << endl;
		ofs << "Intotal: " << InsertTime << endl;
		string queryFilePath = "./query/DMV001";
		cout << queryFilePath << endl;
		// testCardPerformance(CI, queryFilePath);
		cout << "Intotal: " << InsertTime << endl;
		int idx = 0;
		int wrongN = 0;
		int less = 0;
		long long timeSum = 0;
		for (auto ptr = LinkEDList; ptr != NULL; ptr = ptr->Next)
		{
			idx++;
			auto tPStart = Clock::now(); 
			auto page = CardIndexLeafGet(CI, ptr->Key[0]);
			auto tPEnd = Clock::now(); 
			timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(tPEnd - tPStart).count();
		}
		cout << timeSum << endl;
		cout << idx << endl;
		cout << "Avg PQTime:" << timeSum / idx << endl;
	}
}
// transZD("./data/powerOriAllD.txt", "./data/power");
// transZD("./data/DMVintAllD.txt", "./data/DMV");
// transZD("./data/OSMC.txt", "./data/OSM");
// transZD("./data/powerC10", "./data/powerB10");
// transZD("./data/powerC20", "./data/powerB20");
// transZD("./data/powerC95", "./data/powerB95");
// transZD("./data/powerC90", "./data/powerB90");
// transZD("./data/powerC80", "./data/powerB80");
// cout<<"TransDone"<<endl;
int main(int argc, char *argv[])
{
	string jobname = argv[1];
	string dataname = argv[2];
	if (jobname == "CE")
	{
		if (dataname == "power")
		{
			ofs.open("./record/CE_power");
			testFullCardPerformance();
			ofs.close();
		}
		if (dataname == "DMV")
		{
			ofs.open("./record/CE_DMV");
			testFullCardPerformanceDMV();
			ofs.close();
		}
		if (dataname == "OSM")
		{
			ofs.open("./record/CE_OSM");
			testFullCardPerformanceOSM();//The OSM dataset is to large, need to malloc twice.
			ofs.close();
		}
	}
	else if (jobname == "Index")
	{
		if (dataname == "power")
		{
			ofs.open("./record/Index_power");
			testFullRangeQPerformance();
			ofs.close();
		}
		if (dataname == "OSM")
		{
			ofs.open("./record/Index_OSM");
			testFullRangeQPerformanceOSM();//The OSM dataset is to large, need to malloc twice.
			ofs.close();
		}
		if (dataname == "DMV")
		{
			ofs.open("./record/Index_DMV");
			testFullRangeQPerformanceDMV();
			ofs.close();
		}
	}
	return 0;
}