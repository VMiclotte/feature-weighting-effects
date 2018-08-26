import weka.core.Instance;

import java.util.Arrays;

public class Similarity {
    private double[] weights;

    public Similarity(int length){
        weights = new double[length];
        Arrays.fill(weights, 1);
    }

    public double similarity(Instance a, Instance b, double max) {
        double sim = 1 - Math.sqrt(squaredEucDist(a, b)) / max;
        return sim;
    }

    public double squaredEucDist(Instance a, Instance b) {
        int n = a.numAttributes() - 1;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.pow(weights[i] * (a.value(a.attribute(i)) - b.value(a.attribute(i))), 2);
        }
        return sum;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    //Kleene-Dienes implicator
    public double impl(double x, double y) {
        return Math.max(1-x, y);
    }

    public double tNorm(double x, double y) {
        return Math.min(x, y);
    }


    /*public double lowApp(double classValue, Instance x) {
        double inf = 1;
        for(Instance neighbour: neighbours) {
            double val;
            if(neighbour.classValue()==classValue) {
                val = impl(similarity(neighbour, x),1);
            } else {
                val = impl(similarity(neighbour, x),0);
            }
            if(val<inf) {
                inf = val;
            }
        }
        return inf;
    }

    public double uppApp(double classValue, Instance x) {
        double sup = 0;
        for(Instance neighbour: neighbours) {
            double val;
            if(neighbour.classValue()==classValue) {
                val = tNorm(similarity(neighbour, x),1);
            } else {
                val = tNorm(similarity(neighbour, x),0);
            }
            if(val>sup) {
                sup = val;
            }
        }
        return sup;
    }*/

}
