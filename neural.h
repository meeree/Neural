#ifndef  __NEURAL_H__
#define  __NEURAL_H__

#include <armadillo>

#define USE_DOUBLES
#ifdef USE_DOUBLES 
    typedef double FLT;
#else 
    typedef float FLT;
#endif 
typedef arma::Col<FLT> Vec;
typedef arma::Mat<FLT> Mat;

typedef std::vector<unsigned> NetBlueprint;

struct TestResultPair 
{
    Vec inp;
    Vec goal;
};
typedef std::vector<TestResultPair> Batch;

class Network 
{
private:
    unsigned m_nLayers;
    NetBlueprint m_blueprint;
    std::vector<Vec> m_biases; 
    std::vector<Mat> m_weights; 

    void UpdateMiniBatch (Batch& batch, FLT const& rate);
    void Backprop (TestResultPair const& testPair, std::vector<Vec>& biasSkew, std::vector<Mat>& weightSkew);
    static inline Vec CostDerivative (Vec const& activationOutput, Vec const& goalOutput) {return activationOutput-goalOutput;}

public:
    Network (NetBlueprint const& blueprint);

    void Solve (Batch& trainingData, unsigned const& epochs, unsigned const& miniBatchSize, FLT const& rate);
};

#endif //__NEURAL_H__
