public class FQ {
	private double alp; 
	private double bet;
	
	public FQ(double a, double b) {
		alp = a;
		bet = b;
	}
	
  	//Fuzzy quantifiers  	
  	public double fuzQua(double x) {
  		double ret;
  		double c = (alp + bet)/2.0;
  		if(x<= c) {
  			if(x<=alp) {
  				ret = 0;
  			} else {
  				ret = 2*Math.pow((x-alp)/(bet-alp), 2);
  			}
  		} else {
  			if(bet<=x) {
  				ret = 1;
  			} else {
  				ret = 1 - 2*Math.pow((x-alp)/(bet-alp), 2);
  			}
  		}
  		return ret;
  	}
  	
  	
  	
}
