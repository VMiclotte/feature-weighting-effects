import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Created by victo on 24/10/2017.
 */
public class Reader {

    private Instances[] trainDataFolds;
    private Instances[] testDataFolds;
    private int numFolds;
    private String path;

    public Reader(String path, int numFolds){
        trainDataFolds = new Instances[numFolds];
        testDataFolds = new Instances[numFolds];
        this.numFolds = numFolds;
        this.path = path;
    }

    public Instances[] getTrainDataFolds() {
        return trainDataFolds;
    }

    public Instances[] getTestDataFolds() {
        return testDataFolds;
    }

    public void foldsRead() throws Exception {
        for (int fold = 1; fold <= 10; fold++) {
            // read in the data
            String trainFileName = path + "-"+numFolds+"-" + fold + "tra.arff";
            String testFileName = path + "-"+numFolds+"-" + fold + "tst.arff";
            //System.out.println(trainFileName);
            trainDataFolds[fold-1] = DataSource.read(trainFileName);
            testDataFolds[fold-1] = DataSource.read(testFileName);
            trainDataFolds[fold-1].setClassIndex(trainDataFolds[fold-1].numAttributes() - 1);
            testDataFolds[fold-1].setClassIndex(testDataFolds[fold-1].numAttributes() - 1);
        }
    }

    public void foldsReadDiscretized() throws Exception{
        for (int fold = 1; fold <= 10; fold++) {
            // read in the data
            String trainFileName = path + "-d-"+numFolds+"-" + fold + "tra.arff";
            String testFileName = path + "-d-"+numFolds+"-" + fold + "tst.arff";
            System.out.println(trainFileName);
            trainDataFolds[fold-1] = DataSource.read(trainFileName);
            testDataFolds[fold-1] = DataSource.read(testFileName);
            trainDataFolds[fold-1].setClassIndex(trainDataFolds[fold-1].numAttributes() - 1);
            testDataFolds[fold-1].setClassIndex(testDataFolds[fold-1].numAttributes() - 1);
        }
    }
}
