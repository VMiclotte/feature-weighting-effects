
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class UppClassifier extends AbstractClassifier {
    private Instances trainData;
    private int k;
    private double[] weights;
    private List<Double> distances;
    private Map<Double, Instance> similarities;
    private double range;
    private double[] ranges;
    private int numberFeatures;

    public UppClassifier(Instances trainData, int k) {
        numberFeatures = trainData.numAttributes() - 1;
        weights = new double[numberFeatures];
        this.trainData = trainData;
        this.k = k;
        similarities = new HashMap<>();
        distances = new ArrayList<>(trainData.numInstances());
        Arrays.fill(weights, 1);
        range = 0;
        ranges = new double[numberFeatures];
        rangeAttribute();
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] dist = new double[trainData.numClasses()];
        Instance[] n = getNearestNeighbours(instance, k);
        for (int i = 0; i < trainData.numClasses(); i++) {
            dist[i] = upp(n, i, instance);
        }
        return dist;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
    }

    public double squaredEucDist(Instance a, Instance b, double[] w) {
        int n = a.numAttributes() - 1;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.pow(w[i] * (a.value(a.attribute(i)) - b.value(a.attribute(i))), 2);
        }
        return sum;
    }

    public void rangeAttribute() {
        for (int att = 0; att < numberFeatures; att++) {
            Instance first = trainData.instance(0);
            double min = first.value(att);
            double max = first.value(att);
            for (int i = 1; i < trainData.numInstances(); i++) {
                Instance y = trainData.instance(i);
                double value = y.value(att);
                if (value < min) {
                    min = y.value(att);
                }
                if (value > max) {
                    max = y.value(att);
                }
            }
            ranges[att] = max - min;
        }
    }

    public double minDist(Instance a, Instance b, double[] w) {
        double min = 1;
        for (int i = 0; i < numberFeatures; i++) {
            double val = 1 - w[i] * (Math.abs(a.value(i) - b.value(i)) / ranges[i]);
            if (val < min) {
                min = val;
            }
        }
        return min;
    }

    public Instance[] getNearestNeighbours(Instance x, int k) {
        for (int i = 0; i < trainData.numInstances(); i++) {
            Instance y = trainData.instance(i);
            double dist = Math.sqrt(squaredEucDist(x, y, weights));
            distances.add(dist);
            if (range < dist) {
                range = dist;
            }
        }
        for (int i = 0; i < trainData.numInstances(); i++) {
            distances.set(i, 1 - distances.get(i) / range);
            similarities.put(distances.get(i), trainData.instance(i));
        }
        Collections.sort(distances);
        Collections.reverse(distances);
        Instance[] neighbours = new Instance[k];
        for (int i = 0; i < k; i++) {
            neighbours[i] = similarities.get(distances.get(i));
        }
        return neighbours;
    }

    public Instance[] getNearestNeighbours2(Instance x, int k) {
        for (int i = 0; i < trainData.numInstances(); i++) {
            Instance y = trainData.instance(i);
            double dist = minDist(x, y, weights);
            distances.add(dist);
            similarities.put(dist, y);
        }
        Collections.sort(distances);
        Collections.reverse(distances);
        Instance[] neighbours = new Instance[k];
        for (int i = 0; i < k; i++) {
            neighbours[i] = similarities.get(distances.get(i));
        }
        return neighbours;
    }

    public double similarity(Instance a, Instance b) {
        double sim = 1 - Math.sqrt(squaredEucDist(a, b, weights)) / range;
        return sim;
    }

    public double similarity2(Instance a, Instance b) {
        double sim = minDist(a, b, weights);
        return sim;
    }

    public double upp(Instance[] n, double classValue, Instance x) {
        double sum = 0;
        double sum2 = 0;
        for (Instance neighbour : n) {
            if (neighbour.classValue() == classValue) {
                sum += similarity(x, neighbour);
            }
            sum2 += similarity(x, neighbour);
        }
        FQ most = new FQ(0.2, 1);
        double ret = most.fuzQua(sum / sum2);
        return ret;
    }
}