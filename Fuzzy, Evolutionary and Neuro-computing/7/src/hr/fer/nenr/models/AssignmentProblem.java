package hr.fer.nenr.models;

import java.io.File;

import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBVar;

public class AssignmentProblem {
	public static String path = "/home/doms/Desktop/9.SEMESTAR/Operacijska istraživanja/DZ/DZ2/2.lp";
	public static final int OPTIM_MODE = GRB.MAXIMIZE;
	public static final File LOG = new File("./log.txt");

	
	public static int[][] assignmentProblem(double[][] costMatrix) {
		int nVariables = costMatrix.length;
		int[][] results = new int[nVariables][nVariables];
		try {
			GRBEnv env = new GRBEnv();
			GRBModel model = new GRBModel(env);
			GRBVar[][] vars = new GRBVar[nVariables][nVariables];
			for (int i = 0; i < nVariables; ++i) {
				for (int j = 0; j < nVariables; ++j) {
					String varName = String.format("X_%d_%d", i, j);
					vars[i][j] = model.addVar(0, 1, costMatrix[i][j], GRB.BINARY, varName);
				}
			}
			// The objective is to minimize the costs
			model.set(GRB.IntAttr.ModelSense, OPTIM_MODE);

			for (int i = 0; i < nVariables; ++i) {
				String const_name1 = String.format("C_0%d", i);
				String const_name2 = String.format("C_1%d", i);
				GRBLinExpr terms1 = new GRBLinExpr();
				GRBLinExpr terms2 = new GRBLinExpr();
				for (int j = 0; j < nVariables; ++j) {
					terms1.addTerm(1, vars[i][j]);
					terms2.addTerm(1, vars[j][i]);
				}
				model.addConstr(terms1, GRB.EQUAL, 1, const_name1);
				model.addConstr(terms2, GRB.EQUAL, 1, const_name2);
			}
			model.optimize();
			for (int i = 0; i < nVariables; ++i) {
				for (int j = 0; j < nVariables; ++j) {
					results[i][j] = (int) vars[i][j].get(GRB.DoubleAttr.X);
				}
			}
			model.dispose();
			env.dispose();
		} 
		catch (GRBException e) {
			e.printStackTrace();
		}
		return results;
	}

//	public static void main(String[] args) {
//		double[][] cost = { { 14, 5, 8, 7 }, { 2, 12, 6, 5 }, { 7, 8, 3, 9 }, { 2, 4, 6, 10 } };
//		assignmentProblem(cost);
//	}

}
