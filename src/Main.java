import weka.attributeSelection.*;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {
    private static Instances[][] folds; //folds, training and test ([0] and [1] resp.)
    private static int numFolds; //amount of folds
    private static int k; //amount of neighbours
    private static String path;
    private static Writer w = new Writer();
    private static int numAttributes;
    private static int numInstances;
    private static List<double[][]> allWeights = new ArrayList<>();
    private static boolean useNN;

    /*
     * reads folds from files
     */
    public static void readFolds() throws Exception {
        Path p = Paths.get(path);
        String name = p.getFileName().toString();
        Reader r = new Reader(path + "/" + name, numFolds);
        r.foldsRead();
        folds[0] = r.getTrainDataFolds();
        folds[1] = r.getTestDataFolds();
        numAttributes = folds[0][0].numAttributes() - 1;
    }

    public static void readFoldsDiscretized() throws Exception {
        Path p = Paths.get(path);
        String name = p.getFileName().toString();
        Reader r = new Reader(path + "/" + name, numFolds);
        r.foldsReadDiscretized();
        folds[0] = r.getTrainDataFolds();
        folds[1] = r.getTestDataFolds();
        numAttributes = folds[0][0].numAttributes() - 1;
    }

    public static void saveFolds() throws Exception {
        //read the data and randomize
        SimpleReader s = new SimpleReader(path + ".arff");
        s.read();
        Instances dataSet = s.getData();
        System.out.println("inst :" + s.getData().numInstances());
        Random r = new Random();
        r.setSeed(1);
        dataSet.randomize(r);
        numAttributes = dataSet.numAttributes() - 1;
        System.out.println("Att : " + numAttributes);
        //make folds from the data
        Instances[] train = new Instances[numFolds];
        Instances[] test = new Instances[numFolds];
        for (int i = 0; i < numFolds; i++) {
            train[i] = dataSet.trainCV(numFolds, i);
            test[i] = dataSet.testCV(numFolds, i);
        }
        folds[0] = train;
        folds[1] = test;

        /*
         * saves the folds in files in a separate directory with format dataName-numFolds-fold-tra.arff for training data
         * and dataName-numFolds-fold-tst.arff for test data
         */
        Path p = Paths.get(path);
        String name = p.getFileName().toString();
        int pos = name.lastIndexOf(".");
        String justname = pos > 0 ? name.substring(0, pos) : name;
        Path folder = p.getParent().resolve(justname);
        Path trainPath;
        Path testPath;
        File dir = new File(folder.toString());
        dir.mkdir();
        for (int i = 0; i < numFolds; i++) {
            trainPath = folder.resolve(justname + "-" + numFolds + "-" + (i + 1) + "tra.arff");
            testPath = folder.resolve(justname + "-" + numFolds + "-" + (i + 1) + "tst.arff");
            w.writeFold(trainPath, folds[0][i]);
            w.writeFold(testPath, folds[1][i]);
        }
    }

    public static void saveFoldsDiscretized() throws Exception {
        //read the data and randomize
        SimpleReader s = new SimpleReader(path + ".arff");
        s.read();
        Instances dataSet = s.getData();
        Random r = new Random();
        r.setSeed(1);
        dataSet.randomize(r);
        numAttributes = dataSet.numAttributes() - 1;
        //make folds from the data
        Instances[] train = new Instances[numFolds];
        Instances[] test = new Instances[numFolds];
        for (int i = 0; i < numFolds; i++) {
            DiscretizeKMeans dTrain = new DiscretizeKMeans(dataSet.trainCV(numFolds, i));
            dTrain.discretize();
            train[i] = dTrain.getInstances();
            train[i] = dataSet.trainCV(numFolds, i);
            DiscretizeKMeans dTest = new DiscretizeKMeans(dataSet.testCV(numFolds, i));
            dTest.discretize();
            test[i] = dTest.getInstances();
        }
        folds[0] = train;
        folds[1] = test;

        /*
         * saves the folds in files in a separate directory with format dataName-numFolds-fold-tra.arff for training data
         * and dataName-numFolds-fold-tst.arff for test data
         */
        Path p = Paths.get(path);
        String name = p.getFileName().toString();
        int pos = name.lastIndexOf(".");
        String justname = pos > 0 ? name.substring(0, pos) : name;
        Path folder = p.getParent().resolve(justname);
        Path trainPath;
        Path testPath;
        File dir = new File(folder.toString());
        dir.mkdir();
        for (int i = 0; i < numFolds; i++) {
            trainPath = folder.resolve(justname + "-d-" + numFolds + "-" + (i + 1) + "tra.arff");
            testPath = folder.resolve(justname + "-d-" + numFolds + "-" + (i + 1) + "tst.arff");
            w.writeFold(trainPath, folds[0][i]);
            w.writeFold(testPath, folds[1][i]);
        }
    }

    public static void saveFoldsDiscretized(int k) throws Exception {
        //read the data and randomize
        SimpleReader s = new SimpleReader(path + ".arff");
        s.read();
        Instances dataSet = s.getData();
        Random r = new Random();
        r.setSeed(1);
        dataSet.randomize(r);
        numAttributes = dataSet.numAttributes() - 1;
        //make folds from the data
        Instances[] train = new Instances[numFolds];
        Instances[] test = new Instances[numFolds];
        for (int i = 0; i < numFolds; i++) {
            DiscretizeKMeans dTrain = new DiscretizeKMeans(dataSet.trainCV(numFolds, i), k);
            dTrain.discretize();
            train[i] = dTrain.getInstances();
            train[i] = dataSet.trainCV(numFolds, i);
            DiscretizeKMeans dTest = new DiscretizeKMeans(dataSet.testCV(numFolds, i), k);
            dTest.discretize();
            test[i] = dTest.getInstances();
        }
        folds[0] = train;
        folds[1] = test;

        /*
         * saves the folds in files in a separate directory with format dataName-numFolds-fold-tra.arff for training data
         * and dataName-numFolds-fold-tst.arff for test data
         */
        Path p = Paths.get(path);
        String name = p.getFileName().toString();
        int pos = name.lastIndexOf(".");
        String justname = pos > 0 ? name.substring(0, pos) : name;
        Path folder = p.getParent().resolve(justname);
        Path trainPath;
        Path testPath;
        File dir = new File(folder.toString());
        dir.mkdir();
        for (int i = 0; i < numFolds; i++) {
            trainPath = folder.resolve(justname + "-d-" + numFolds + "-" + (i + 1) + "tra.arff");
            testPath = folder.resolve(justname + "-d-" + numFolds + "-" + (i + 1) + "tst.arff");
            w.writeFold(trainPath, folds[0][i]);
            w.writeFold(testPath, folds[1][i]);
        }
    }

    public static double[] classify(double[] weights, int fold) throws Exception {
        /*System.out.println("Weights: ");
        System.out.print(weights[0]);
        for(int i = 1; i < weights.length; i++){
            System.out.print(", " + weights[i]);
        }
        System.out.println();*/
        double[] result = new double[6];
        Instances trainData = folds[0][fold];
        Instances testData = folds[1][fold];
        if (!useNN) {
            k = trainData.numInstances();
        }


        //System.out.println("number of classes in fold " + fold + " : " + testData.numClasses());
        // class accuracies storage
        double[] accs_low = new double[testData.numClasses()];
        double[] accs_av = new double[testData.numClasses()];
        double[] accs_upp = new double[testData.numClasses()];
        double[] counts = new double[testData.numClasses()];

        // classification of the test data
        for (int i = 0; i < testData.numInstances(); i++) {
            Instance x = testData.instance(i);
            counts[(int) x.classValue()]++;

            LowClassifier low = new LowClassifier(trainData, k);
            AverageClassifier av = new AverageClassifier(trainData, k);
            UppClassifier upp = new UppClassifier(trainData, k);

            low.setWeights(weights);
            av.setWeights(weights);
            upp.setWeights(weights);

            double pred_low = low.classifyInstance(x);
            if (pred_low == x.classValue()) {
                accs_low[(int) pred_low]++;
            }

            double pred_av = av.classifyInstance(x);
            if (pred_av == x.classValue()) {
                accs_av[(int) pred_av]++;
            }

            double pred_upp = upp.classifyInstance(x);
            if (pred_upp == x.classValue()) {
                accs_upp[(int) pred_upp]++;
            }

                /*System.out.println("Fold = " + fold);
                System.out.println("SimpleReader object = " + x);
                System.out.println("Actual class of test object = " + x.classValue());
                System.out.println("Prediction with Low Classifier of x = " + pred_low);*/
        }

        // accuracy calculation

        double acc_low = 0;
        //System.out.println("===accs low");
        for (int cl = 0; cl < accs_low.length; cl++) {
            //System.out.println(accs_low[cl]);
            acc_low += accs_low[cl];
        }
        //System.out.println("numInstances in testData  " + testData.numInstances());
        result[0] += acc_low / testData.numInstances();

        double acc_av = 0;
        //System.out.println("===accs av");
        for (int cl = 0; cl < accs_av.length; cl++) {
            //System.out.println(accs_av[cl]);
            acc_av += accs_av[cl];
        }
        result[1] += acc_av / testData.numInstances();

        double acc_upp = 0;
        //System.out.println("===accs upp");
        for (int cl = 0; cl < accs_upp.length; cl++) {
            //System.out.println(accs_upp[cl]);
            acc_upp += accs_upp[cl];
        }
        result[2] += acc_upp / testData.numInstances();

        // balanced accuracy calculation
        int count = 0;

        double bal_acc_low = 0;
        for (int cl = 0; cl < accs_low.length; cl++) {
            if (counts[cl] > 0) {
                bal_acc_low += accs_low[cl] / counts[cl];
                count++;
            }
        }
        bal_acc_low /= count;
        result[3] += bal_acc_low;

        double bal_acc_av = 0;
        for (int cl = 0; cl < accs_av.length; cl++) {
            if (counts[cl] > 0) {
                bal_acc_av += accs_av[cl] / counts[cl];
            }
        }
        bal_acc_av /= count;
        result[4] += bal_acc_av;

        double bal_acc_upp = 0;
        for (int cl = 0; cl < accs_upp.length; cl++) {
            if (counts[cl] > 0) {
                bal_acc_upp += accs_upp[cl] / counts[cl];
            }
        }
        bal_acc_upp /= count;
        result[5] += bal_acc_upp;
        return result;
    }

    public static void classifyTest() throws Exception {

        System.out.println("Accuracy without weight vector");
        classify_CV();
        System.out.println();

        System.out.println("Accuracy with weight vector (InfGain)");
        classify_CV_weightsInfGain();
        System.out.println();

        System.out.println("Accuracy with weight vector (InfGain ratio)");
        classify_CV_weightsInfGainRatio();
        System.out.println();

        System.out.println("Accuracy with weight vector (Symmetrical Uncertainty)");
        classify_CV_weightsSymmUncert();
        System.out.println();

        System.out.println("Accuracy with weight vector (Kullback)");
        classify_CV_weightsKB();
        System.out.println();

        System.out.println("Accuracy with weight vector (Laplace smoothed Kullback)");
        classify_CV_weightsKBL();
        System.out.println();

        System.out.println("Accuracy with weight vector (SVM method)");
        classify_CV_weightsSVM();
        System.out.println();

        System.out.println("Accuracy with weight vector (Relief method)");
        classify_CV_weightsRelief();
        System.out.println();

        System.out.println("Accuracy with weight vector (One Rules)");
        classify_CV_weightsOneR();
        System.out.println();

        System.out.println("Accuracy with ensemble weight vector");
        classify_CV_ensemble();
        System.out.println();
    }

    public static void main(String[] args) throws Exception {
        path = "Data/heart";
        numFolds = 10;
        k = 3;
        useNN = true;
        boolean discretize = false;
        boolean makeFolds = false;

        folds = new Instances[2][numFolds];

        if (discretize) {
            if (makeFolds) {
                saveFoldsDiscretized();//makes cross-validation folds and saves them into files using kmeans to discretize every attribute with k equal to the amount of classes
            }
            readFoldsDiscretized(); //reads the cross-validation folds
        } else {
            if (makeFolds) {
                saveFolds();//makes cross-validation folds and saves them into files
            }
            readFolds(); //reads the cross-validation folds
        }
        classifyTest();
        Path p = Paths.get(path);
    }

    public static void classify_CV() throws Exception {
        double[] result = new double[6];

        for (int fold = 0; fold < numFolds; fold++) {
            double[] weights = new double[numAttributes];
            Arrays.fill(weights, 1.0);
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }

        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsInfGain() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];
            InfoGainAttributeEval ev = new InfoGainAttributeEval();
            ev.buildEvaluator(trainData);
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                weights[i] = ev.evaluateAttribute(i);
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsInfGainRatio() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];
            GainRatioAttributeEval ev = new GainRatioAttributeEval();
            ev.buildEvaluator(trainData);
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                weights[i] = ev.evaluateAttribute(i);
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsSymmUncert() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];
            SymmetricalUncertAttributeEval ev = new SymmetricalUncertAttributeEval();
            ev.buildEvaluator(trainData);
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                weights[i] = ev.evaluateAttribute(i);
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);

        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsKB() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Weights w = new Weights(folds[0][fold]);
            double[] weights = w.kbWeight();
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);

        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsKBL() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Weights w = new Weights(folds[0][fold]);
            double[] weights = w.kbWeightL();
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsSVM() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            WeightsSVM w = new WeightsSVM(folds[0][fold]);
            double[] weights = w.getWeightsMax();
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsRelief() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];
            ReliefFAttributeEval ev = new ReliefFAttributeEval();
            ev.buildEvaluator(trainData);
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                weights[i] = ev.evaluateAttribute(i);
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_weightsOneR() throws Exception {
        double[] result = new double[6];
        double[][] foldWeights = new double[numFolds][folds[0][0].numAttributes() - 1];
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];
            OneRAttributeEval ev = new OneRAttributeEval();
            ev.buildEvaluator(trainData);
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                weights[i] = ev.evaluateAttribute(i);
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
            foldWeights[fold] = weights;
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        allWeights.add(foldWeights);
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }

    public static void classify_CV_ensemble() throws Exception {
        double[] result = new double[6];
        int numMethods = allWeights.size();
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainData = folds[0][fold];
            double[] weights = new double[trainData.numAttributes() - 1];

            for (int i = 0; i < numMethods; i++) {
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                //normalizing weights
                for (int j = 0; j < weights.length; j++) {
                    if (allWeights.get(i)[fold][j] < min) {
                        min = allWeights.get(i)[fold][j];
                    }
                    if (allWeights.get(i)[fold][j] > max) {
                        max = allWeights.get(i)[fold][j];
                    }
                }
                //taking average of weight vector across all weighting methods
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += (allWeights.get(i)[fold][j] - min) / (numMethods * (max - min));
                }
            }
            double[] accuracies = classify(weights, fold);
            for (int i = 0; i < 6; i++) {
                result[i] += accuracies[i];
            }
        }
        for (int i = 0; i < 6; i++) {
            result[i] = result[i] / numFolds;
        }
        // print out the results
        System.out.println("Accuracy_low : " + result[0]);
        System.out.println("Balanced_accuracy_low : " + result[1]);
        System.out.println("Accuracy_av : " + result[2]);
        System.out.println("Balanced_accuracy_av : " + result[3]);
        System.out.println("Accuracy_upp : " + result[4]);
        System.out.println("Balanced_accuracy_upp : " + result[5]);

        // empty line
        System.out.println();
        w.add(result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ", " + result[4] + ", " + result[5]);
    }
}