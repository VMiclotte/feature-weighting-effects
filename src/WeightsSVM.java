import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.Arrays;

public class WeightsSVM {
    private double[] weights;
    private Instances train;
    private int numAttributes;
    private int numClasses;
    private SMO smo = new SMO();
    private double[][][] svmWeights;

    public WeightsSVM(Instances trainData) throws Exception {
        smo.buildClassifier(trainData);
        numAttributes = trainData.numAttributes()-1;
        numClasses = trainData.numClasses();
        weights = new double[numAttributes];
        train = trainData;
        svmWeights = smo.sparseWeights();
    }

    public double[] getWeightsAv(){
        int count= 0;
        double[] sumOfWeights = new double[numAttributes];
        for (int i = 0; i < numClasses - 1; i++) {
            for (int j = i + 1; j < numClasses; j++) {
                if (svmWeights[i][j]!= null && svmWeights[i][j].length == numAttributes) {
                    for (int k = 0; k < numAttributes; k++) {
                        sumOfWeights[k] += Math.abs(svmWeights[i][j][k]);
                    }
                    count++;
                }
            }
        }
        for (int u = 0; u < numAttributes; u++) {
            weights[u] += sumOfWeights[u] / count;
        }
        return weights;
    }

    public double[] getWeightsMax(){
        for (int i = 0; i < numClasses - 1; i++) {
            for (int j = i + 1; j < numClasses; j++) {
                if (svmWeights[i][j]!= null && svmWeights[i][j].length == numAttributes) {
                    for (int k = 0; k < numAttributes; k++) {
                        double weight = svmWeights[i][j][k];
                        if(Math.abs(weights[k])<Math.abs(weight)){
                            weights[k] = weight;
                        }
                    }
                }
            }
        }
        return weights;
    }
}
