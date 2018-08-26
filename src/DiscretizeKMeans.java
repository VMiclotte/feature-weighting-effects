import Clustering.KMeans;
import Clustering.Point;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class DiscretizeKMeans {
    private Instances instances;
    private int n;
    private int numInstances;
    private int numAttributes;
    public DiscretizeKMeans(Instances instances, int n){
        this.instances = instances;
        this.n = n;
        numAttributes = instances.numAttributes() - 1;
        numInstances = instances.numInstances();
    }
    public DiscretizeKMeans(Instances instances){
        this.instances = instances;
        this.n = instances.numClasses();
        numAttributes = instances.numAttributes() - 1;
        numInstances = instances.numInstances();
    }


    public double[] kMeansCenters(int attribute){
        double[] values = instances.attributeToDoubleArray(attribute);
        ArrayList<Point> points = new ArrayList<>();
        for(double d : values){
            points.add(new Point(new double[]{d}));
        }
        KMeans kMeans = new KMeans(points,n);
        Point[] centers = kMeans.cluster();
        double[] result = new double[n];
        for(int i = 0; i < n; i++){
            result[i] = centers[i].getCoord(0);
        }
        return result;
    }

    public void discretize(){
        double[][] centers = new double[numAttributes][n];
        for(int i = 0; i < numAttributes; i++){
            centers[i] = kMeansCenters(i);
        }
        for(int i = 0; i < numInstances; i++){
            Instance original = instances.get(i);
            Instance instance = (Instance) original.copy();
            instance.setClassValue(original.classValue());
            for(int j = 0; j < numAttributes; j++){
                double val = original.value(j);
                int k = 0;
                double newValue = closestCenter(centers[j],val);
                instance.setValue(j,newValue);
            }
            instances.add(i,instance);
            instances.remove(i+1);
        }
    }

    public double dist(double a, double b){
        return Math.abs(a-b);
    }

    public double closestCenter(double[] centers, double value){
        double minDist = dist(centers[0],value);
        int index = 0;
        for(int i = 1; i < n; i++){
            double newDist = dist(centers[i],value);
            if(minDist>newDist){
                minDist = newDist;
                index = i;
            }
        }
        return centers[index];
    }


    public Instances getInstances() {
        return instances;
    }
}
