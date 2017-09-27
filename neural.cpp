#include "neural.h"
#include <algorithm>
#include <iostream>
#include <fstream>

#define ASSERT_VOID(cnd, errmsg...) if(!(cnd)) {printf(errmsg); return;}
#define ASSERT(cnd, err_ret, errmsg...) if(!(cnd)) {printf(errmsg); return err_ret;}

inline Vec sig (Vec const& z) 
{
    return 1./(1.+arma::exp(-z));
}

inline Vec sigp (Vec const& z)
{
    return 1./(2.-(arma::exp(z)+arma::exp(-z)));
}

Network::Network (NetBlueprint const& blueprint)
    : m_blueprint{blueprint}, m_nLayers{(unsigned)blueprint.size()}
{
    ASSERT_VOID(m_nLayers > 0, "Attempt to create empty network");
    m_biases.resize(m_nLayers-1);
    m_weights.resize(m_nLayers-1);
    for(unsigned i = 0, j = 1; j < m_nLayers; ++i, ++j)
    {
        m_biases[i].resize(m_blueprint[j]);
        m_weights[i].resize(m_blueprint[j], m_blueprint[i]);
    }
}

void Network::Solve (Batch& trainingData, unsigned const& epochs, unsigned const& miniBatchSize, FLT const& rate)
{
    size_t const& n{trainingData.size()};
    size_t const  m{n/miniBatchSize}; //Note that we will lose n % m data points if ¬(m | n)
    
    unsigned const curTime = std::chrono::system_clock::now().time_since_epoch().count();
    for(unsigned epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(trainingData.begin(), trainingData.end(), std::default_random_engine(curTime));

        std::vector<Batch> miniBatches(m);
        unsigned i{0};
        for(auto batchIt = trainingData.begin(); batchIt != trainingData.end(); batchIt += miniBatchSize, ++i)
        {
            miniBatches[i] = Batch(batchIt, batchIt + miniBatchSize);
        }
        for(auto& mini: miniBatches)
        {
            UpdateMiniBatch(mini, rate);
        }
    }
}

void Network::UpdateMiniBatch(Batch& mini, FLT const& rate)
{
    std::vector<Vec> biasSkew(m_nLayers-1);
    std::vector<Mat> weightSkew(m_nLayers-1);
    for(unsigned i = 0; i < m_nLayers-1; ++i)
    {
        biasSkew[i].copy_size(m_biases[i]);
        biasSkew[i].zeros();
        weightSkew[i].copy_size(m_weights[i]);
        weightSkew[i].zeros();
    }

    for(auto const& testPair: mini)
    {
        Backprop(testPair, biasSkew, weightSkew);
    }

    for(unsigned i = 0; i < m_nLayers-1; ++i)
    {
        m_weights[i] -= (rate/mini.size()) * weightSkew[i];
        m_biases[i] -= (rate/mini.size()) * biasSkew[i];
    }
}

void Network::Backprop (TestResultPair const& testPair, std::vector<Vec>& biasSkew, std::vector<Mat>& weightSkew)
{

    //Feed forward 
    std::vector<Vec> activations{testPair.inp};
    std::vector<Vec> weightedInps; 

    for(unsigned i = 0; i < m_nLayers-1; ++i)
    {
        Mat& w{m_weights[i]};               //Weight matrix
        Vec& b{m_biases[i]};                //Bias vector
        Vec& a{activations.back()};         //Curent activations
        weightedInps.push_back(w*a + b);    //Weighted input to neuron

        Vec& z{weightedInps.back()};        //Newly created weight input
        activations.push_back(sig(z));      //New (output) activations
    }

    //Backward pass
    Vec& a{activations.back()};
    Vec& z{weightedInps.back()}; 
    Vec delta{CostDerivative(a, testPair.goal) * sigp(z)};

    biasSkew[(m_nLayers-1)-1] += delta;
    weightSkew[(m_nLayers-1)-1] += delta * activations[m_nLayers-2].t();
    
    for(unsigned lOpp = 2; lOpp < m_nLayers; ++lOpp)
    {
        z = weightedInps[(m_nLayers-1)-lOpp];
        Mat& w{m_weights[(m_nLayers-1)-lOpp]};
        a = activations[m_nLayers-1-lOpp];

        delta = w.t() * delta * sigp(z);
        biasSkew[(m_nLayers-1)-lOpp] += delta;
        weightSkew[(m_nLayers-1)-lOpp] += delta * a.t();
    }
}

int main ()
{
    std::ifstream labelFl("./train-labels.idx1-ubyte", std::ios::in | std::ios::binary), 
                  imageFl("./train-images.idx3-ubyte", std::ios::in | std::ios::binary);

    int32_t nLabels, nImages, nRows, nCols; 
    labelFl.seekg(4);
    imageFl.seekg(4);
    labelFl.read(reinterpret_cast<char*>(nLabels), sizeof(unsigned char)*4);
    std::cout<<nLabels<<std::endl;
    imageFl.read(reinterpret_cast<char*>(nImages), 4);
    imageFl.read(reinterpret_cast<char*>(nRows), 4);
    imageFl.read(reinterpret_cast<char*>(nCols), 4);
    std::vector<unsigned char> labels(nLabels), images(nImages*nRows*nCols);
    labelFl.read(reinterpret_cast<char*>(labels.data()), nLabels*sizeof(unsigned char));
    imageFl.read(reinterpret_cast<char*>(images.data()), nImages*nRows*nCols*sizeof(unsigned char));
}
