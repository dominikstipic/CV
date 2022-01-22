package hr.fer.tel.rassus.main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.github.plot.Plot;
import com.perfdynamics.pdq.Job;
import com.perfdynamics.pdq.Methods;
import com.perfdynamics.pdq.Node;
import com.perfdynamics.pdq.PDQ;
import com.perfdynamics.pdq.QDiscipline;

public class DZ {
	public static final double[] S = {0.003, 0.001, 0.01, 0.04, 0.1, 0.13, 0.15};
	public static final double[] V = {1.0000, 1.2195, 0.3762, 0.4411, 0.6098, 0.3659, 0.8536};
	
	
	public static void createServers(PDQ pdq) {
		for(int i = 0; i < S.length; ++i) {
	        pdq.CreateNode("S"+i, Node.CEN, QDiscipline.FCFS);
		}
	}
	
	public static void connect(PDQ pdq) {
		for(int i = 0; i < S.length; ++i) {
			pdq.SetVisits("S"+i, "Ulaz", V[i], S[i]);
		}
	}
	
	public static List<Double> linspace(double upper, int n) {
		double delta = upper/n;
		List<Double> deltas = new ArrayList<>();
		for(int i = 1; i <= n; ++i) {
			double x = delta*i;
			deltas.add(x);
		}
		return deltas;
	}
	
	public static void linePlot(List<Double> xs, List<Double> ys) {
		Plot plot = Plot.plot(null).series(null, Plot.data().xy(xs, ys), null);
		try {
			plot.save("./repo/pic", "png");
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
	
	public static void main(String[] args) {
		double lambdaMax = 7.5;
		List<Double> lambdas = linspace(lambdaMax, 50);
		List<Double> time = new ArrayList<>();
		for(double lambda : lambdas) {
			PDQ pdq = new PDQ();
        	pdq.Init("Simulacija");
            pdq.CreateOpen("Ulaz", lambda);
            
            createServers(pdq);
            connect(pdq);
            pdq.Solve(Methods.CANON);
            
            double T = pdq.GetResponse(Job.TRANS, "Ulaz");
            System.out.println(String.format("Lambda = %.3f, T = %.3f", lambda, T));
            time.add(T);
		}
        linePlot(lambdas, time);
	}
}
