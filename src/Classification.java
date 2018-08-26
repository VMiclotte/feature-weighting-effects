import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Classification {

	private String[] dataNames;
	private String[] paths;
	
	
	// set the datasets
	public void setDatasets(){
		dataNames = new String[2];
		paths = new String[2];
		
		// Appendicitis
		dataNames[0] = "Appendicitis";
	    paths[0] = "D:/Victor/Documents/Ugent/Machine learning/Project/Data/appendicitis/appendicitis";
 	 	    
 	 	// Iris
 		dataNames[1] = "Iris";
 	    paths[1] = "D:/Victor/Documents/Ugent/Machine learning/Project/Data/iris/iris";
	}

	/*
	 * Inspection of the datasets
	 */
	public void inspect() throws Exception{
		setDatasets();
		
		int numFolds = 10;
		
		// go through all datasets
		for(int d = 0; d < dataNames.length; d++){
			
			System.out.println("Dataset: " + dataNames[d]);
			
			// run through all folds
			for(int fold = 1; fold <= numFolds; fold++){
				// read in the data
				String trainFileName = paths[d] + "-10-" + fold + "tra.arff";
		        String testFileName =  paths[d] + "-10-" + fold + "tst.arff";
		        Instances trainData = DataSource.read(trainFileName);
		        Instances testData = DataSource.read(testFileName);
		        trainData.setClassIndex(trainData.numAttributes() - 1);
		        testData.setClassIndex(testData.numAttributes() - 1);
		        
		        // print some information
		        System.out.println("Fold " + fold);
		        System.out.println("nTrainInst: " + trainData.numInstances());
		        System.out.println("nTestInst: " + testData.numInstances());
		        System.out.println("nAttrs: " + (trainData.numAttributes() - 1));
		        System.out.println("First five training instances:");
		        for(int i = 0; i < 5; i++){
		        	Instance x = trainData.instance(i);
		        	System.out.println(x);
		        }
		        
		        // empty line
				System.out.println();
			}
			
			// empty line
			System.out.println();
		}
	}
	
	/*
	 * Classification with two classifiers.
	 * Folds for 10-fold cross-validation are provided.
	 */
	public void classify_CV() throws Exception{
		setDatasets();
		int numFolds = 10;
		
		// go through all datasets
		for(int d = 0; d < dataNames.length; d++){
			
			System.out.println("Dataset: " + dataNames[d]);
			
			double total_bal_acc_J48 = 0;
			double total_bal_acc_IBk = 0;
			double total_acc_J48 = 0;
			double total_acc_IBk = 0;
			
			// perform classification for all folds
			for(int fold = 1; fold <= numFolds; fold++){
				// read in the data
				String trainFileName = paths[d] + "-10-" + fold + "tra.arff";
		        String testFileName =  paths[d] + "-10-" + fold + "tst.arff";
		        Instances trainData = DataSource.read(trainFileName);
		        Instances testData = DataSource.read(testFileName);
		        trainData.setClassIndex(trainData.numAttributes() - 1);
		        testData.setClassIndex(testData.numAttributes() - 1);
		        
		        // class accuracies storage
		        double[] accs_J48 = new double[testData.numClasses()];
		        double[] accs_IBk = new double[testData.numClasses()];
		        double[] counts = new double[testData.numClasses()];
		        
		        // train classifiers on the training data		
		        J48 c_J48 = new J48();
		        c_J48.buildClassifier(trainData);
			    IBk c_IBk = new IBk(10);   
			    c_IBk.buildClassifier(trainData);
			    
			    // classification of the test data
			    for(int i = 0; i < testData.numInstances(); i++){
			    	Instance x = testData.instance(i);
			    	counts[(int) x.classValue()]++;		
		    	
			    	// J48
			    	double pred_J48 = c_J48.classifyInstance(x);
			    	if(pred_J48 == x.classValue()){
			    		accs_J48[(int) pred_J48]++;
			    	}
			    	
			    	// IBk
			    	double pred_IBk = c_IBk.classifyInstance(x);
			    	if(pred_IBk == x.classValue()){
			    		accs_IBk[(int) pred_IBk]++;
			    	}
			    }
			    
			    // accuracy calculation
			    double acc_J48 = 0;
			    for(int cl = 0; cl < accs_J48.length; cl++){
			    	acc_J48 += accs_J48[cl];
			    }
			    total_acc_J48 += acc_J48 / testData.numInstances();
			    
			    double acc_IBk = 0;
			    for(int cl = 0; cl < accs_IBk.length; cl++){
			    	acc_IBk += accs_IBk[cl];
			    }
			    total_acc_IBk += acc_IBk / testData.numInstances();			    		    
			    
			    // balanced accuracy calculation
			    double bal_acc_J48 = 0;
			    int count = 0;
			    for(int cl = 0; cl < accs_J48.length; cl++){
			    	if(counts[cl] > 0){
			    		accs_J48[cl] /= counts[cl];	
			    		bal_acc_J48 += accs_J48[cl];
			    		count++;
			    	}			    	
			    }
			    bal_acc_J48 /= count;			    
			    total_bal_acc_J48 += bal_acc_J48;	
			    
			    double bal_acc_IBk = 0;
			    for(int cl = 0; cl < accs_IBk.length; cl++){
			    	if(counts[cl] > 0){
			    		accs_IBk[cl] /= counts[cl];	
			    		bal_acc_IBk += accs_IBk[cl];
			    	}			    	
			    }
			    bal_acc_IBk /= count;			    
			    total_bal_acc_IBk += bal_acc_IBk;
			}
			
			// divide measures by the number of folds
			total_acc_J48 /= numFolds;
			total_acc_IBk /= numFolds;
			total_bal_acc_J48 /= numFolds;
			total_bal_acc_IBk /= numFolds;
			
			// print out the results
			System.out.println("Accuracy_J48 " + dataNames[d] + ": " + total_acc_J48);
			System.out.println("Balanced_accuracy_J48 " + dataNames[d] + ": " + total_bal_acc_J48);
			System.out.println("Accuracy_IBk " + dataNames[d] + ": " + total_acc_IBk);
			System.out.println("Balanced_accuracy_IBk " + dataNames[d] + ": " + total_bal_acc_IBk);
			
			// empty line
			System.out.println();
		}
	}		

	public static void main(String[] args) throws Exception {
		Classification c = new Classification();
		c.inspect();
		c.classify_CV();
	}
}
