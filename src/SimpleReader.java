

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class SimpleReader {
    private Instances data;
    private String path;

    public SimpleReader(String path){
        this.path = path;
    }

    public Instances getData() {
        return data;
    }

    public void read() throws Exception {
        data = ConverterUtils.DataSource.read(path);
        data.setClassIndex(data.numAttributes()-1);
    }

    public Instances[][] makeFolds(Instances data, int numFolds) throws Exception {
        Instances[] train = new Instances[numFolds];
        Instances[] test = new Instances[numFolds];

        for (int i = 0; i < numFolds; i++) {
            train[i] = data.trainCV(numFolds, i);
            test[i] = data.testCV(numFolds, i);
        }
        Instances[][] folds = new Instances[2][numFolds];
        folds[0] = train;
        folds[1] = test;
        return folds;
    }

    public void save(){

    }

}
