
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.beans.Classifier;

import java.util.ArrayList;
import java.util.List;

public class Algoritme1 extends Classifier {
	
	private Instance test;
	private Instances trainData;
	private int k;
	private Instance[] neighbours;
	private Similarity sim;
	private List distances;
	private double range;
	
	public Algoritme1(Instances trainData, int k) {
		this.k = k;
		this.trainData = trainData;
		range = 0;
		distances = new ArrayList<Double>();
	}

	public void setRange(Instance x){
		for (int i = 0; i < trainData.numInstances(); i++) {
			Instance y = trainData.instance(i);
			double dist = Math.sqrt(sim.squaredEucDist(x, y));
			distances.add(dist);
			if (range < dist) {
				range = dist;
			}
		}
	}

	public double classifyInstance(Instance x) {
		int max = 0;
		double cat = 0;
		for(int i = 0; i < trainData.numClasses(); i++) {
			int sum = 0;
			setRange(x);
			for(int j = 0; j < k; j++) {
				Instance n = neighbours[j];
				if(n.classValue()==i) {
					sum += sim.similarity(x, n, range);
				}
			}
			if(sum > max) {
				max = sum;
				cat = i;
			}
		}
		return cat;
	}
}
