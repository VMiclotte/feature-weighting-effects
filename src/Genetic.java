import weka.classifiers.bayes.net.search.global.GeneticSearch;

import java.util.*;

public class Genetic {

    static final long serialVersionUID = 4236165533882462203L;
    int m_nRuns = 10;
    int m_nPopulationSize = 10;
    int m_nDescendantPopulationSize = 100;
    boolean m_bUseCrossOver = true;
    boolean m_bUseMutation = true;
    boolean m_bUseTournamentSelection = false;
    int m_nSeed = 1;
    static boolean[] g_bIsSquare;

    private List best;
    private Random r;
    private Map<Double, double[]> mapping;

    public Genetic(int length){
        best = new ArrayList<double[]>();
        r = new Random();
        for(int i = 0; i < 10; i++){
            double[] nieuw = new double[length];
            for(int j = 0; j < length; j++){
                nieuw[j] = r.nextDouble();
            }
            best.add(nieuw);
        }
        mapping = new HashMap<>();
        for(int i = 0; i < 10; i++){
            mapping.put(0.0, (double[]) best.get(i));
        }

    }

    public List getBest() {
        return best;
    }

    public double[] mutation(double [] v, double value){
        int n = v.length;
        Random r = new Random();
        for(int i = 0; i < n/5; i++){
            int change = r.nextInt(n);
            v[change] += value;
        }
        return v;
    }

    public List crossover(double[] v, double[] w){
        Random r = new Random();
        int n = v.length;
        int index = r.nextInt(n);
        double[] vnew = new double[n];
        double[] wnew = new double[n];
        for(int i = 0; i < index; i++){
            vnew[i] = v[i];
            wnew[i] = w[i];
        }
        for(int i = index; i < n; i++){
            vnew[i] = w[i];
            wnew[i] = v[i];
        }
        List list = new ArrayList<double[]>();
        list.add(vnew);
        list.add(wnew);
        return list;
    }

    public List haveBirth(){
        double[] one = (double[]) best.get(r.nextInt(10));
        double[] two = (double[]) best.get(r.nextInt(10));
        List children = crossover(one, two);
        double[] three = (double[]) best.get(r.nextInt(10));
        children.add(mutation(three, 0.1));
        return children;
    }


    /*public void add(List list){
        for (double[] w :
                list) {
            if(){

            }
        }
    }*/

}
