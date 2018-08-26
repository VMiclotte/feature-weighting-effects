import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;


// volgens Kullback-Leibler measure
public class Weights {

    private double[] weights;
    private Instances train;
    private int numAttributes;

    public Weights(Instances trainData) {
        weights = null;
        train = trainData;
        numAttributes = trainData.numAttributes()-1;
    }

    public double[] getWeights() {
        return weights;
    }

    public double kbMeasure(int featureIndex, double value) {
        int size = train.size();
        Map<Double, Double> values = new HashMap<>();
        Map<Double, Double> classes = new HashMap<>();
        Map<Double[], Double> classvalPairs = new HashMap<>();
        for (Instance i : train) {
            double val = i.value(featureIndex);
            double classval = i.classValue();
            Double[] classvalPair = {val, classval};
            if (values.containsKey(val)) {
                values.put(val, values.get(val) + 1.0);
            } else {
                values.put(val, 1.0);
            }
            if (classes.containsKey(classval)) {
                classes.put(classval, classes.get(classval) + 1.0);
            } else {
                classes.put(classval, 1.0);
            }
            boolean added = false;
            for(Double[] cvPair : classvalPairs.keySet()){
                if(cvPair[0]==val && cvPair[1]==classval){
                    classvalPairs.put(cvPair,classvalPairs.get(cvPair)+1.0);
                    added =true;
                }
            }
            if(!added){
                classvalPairs.put(classvalPair, 1.0);
            }
        }
        double sum = 0;
        for(Double[] classvalPair : classvalPairs.keySet()){
            if(classvalPair[0]==value){
                double probCGivenA = classvalPairs.get(classvalPair) / values.get(value);
                double probC = classes.get(classvalPair[1]) / size;
                sum += (probCGivenA) * Math.log((probCGivenA) / (probC));
            }
        }
        return sum;
    }

    public double kbWeight(int featureIndex) {
        double sum = 0;
        int size = train.size();
        Map<Double,Double> values = new HashMap<>();
        for (Instance i : train) {
            double val = i.value(featureIndex);
            if (values.containsKey(val)) {
                values.put(val, values.get(val) + 1.0);
            } else {
                values.put(val, 1.0);
            }
        }
        for(int j = 0; j < size; j++){
            double val = train.get(j).value(featureIndex);
            sum += values.get(val)/size*kbMeasure(featureIndex,val);
        }
        return sum;
    }


    public double[] kbWeight(){
        double[] w = new double[numAttributes];
        for (int f = 0; f < numAttributes; f++) {
            w[f] = kbWeight(f);
        }
        return w;
    }


    public double kbMeasureL(int featureIndex, double value) {
        int size = train.size();
        Map<Double, Double> values = new HashMap<>();
        Map<Double, Double> classes = new HashMap<>();
        Map<Double[], Double> classvalPairs = new HashMap<>();
        for (Instance i : train) {
            double val = i.value(featureIndex);
            double classval = i.classValue();
            Double[] classvalPair = {val, classval};
            if (values.containsKey(val)) {
                values.put(val, values.get(val) + 1.0);
            } else {
                values.put(val, 1.0);
            }
            if (classes.containsKey(classval)) {
                classes.put(classval, classes.get(classval) + 1.0);
            } else {
                classes.put(classval, 1.0);
            }
            boolean added = false;
            for(Double[] cvPair : classvalPairs.keySet()){
                if(cvPair[0]==val && cvPair[1]==classval){
                    classvalPairs.put(cvPair,classvalPairs.get(cvPair)+1.0);
                    added =true;
                }
            }
            if(!added){
                classvalPairs.put(classvalPair, 1.0);
            }
        }
        double sum = 0;
        for(Double[] classvalPair : classvalPairs.keySet()){
            if(classvalPair[0]==value){
                double probCGivenA = (classvalPairs.get(classvalPair)+1) / (values.get(value)+classes.keySet().size());
                double probC = (classes.get(classvalPair[1])+1) / (size+classes.keySet().size());
                sum += (probCGivenA) * Math.log((probCGivenA) / (probC));
            }
        }
        return sum;
    }

    public double kbWeightL(int featureIndex) {
        double sum = 0;
        int size = train.size();
        Map<Double,Double> values = new HashMap<>();
        for (Instance i : train) {
            double val = i.value(featureIndex);
            if (values.containsKey(val)) {
                values.put(val, values.get(val) + 1.0);
            } else {
                values.put(val, 1.0);
            }
        }
        for(int j = 0; j < size; j++){
            double val = train.get(j).value(featureIndex);
            sum += values.get(val)/size*kbMeasureL(featureIndex,val);
        }
        return sum;
    }


    public double[] kbWeightL(){
        double[] w = new double[numAttributes];
        for (int f = 0; f < numAttributes; f++) {
            w[f] = kbWeightL(f);
        }
        return w;
    }



}
