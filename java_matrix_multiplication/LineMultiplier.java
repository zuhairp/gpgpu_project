/* Modified from https://martin-thoma.com/part-iii-matrix-multiplication-on-multiple-cores-in-python-java-and-c/ */

import java.util.ArrayList;
import java.util.concurrent.Callable;

public class LineMultiplier implements Callable<float[][]> {
	ArrayList<ArrayList<Float>> A;
	ArrayList<ArrayList<Float>> B;
	int start;
	int end;
	public float[][] C;

	public LineMultiplier(ArrayList<ArrayList<Float>> a,
			ArrayList<ArrayList<Float>> b, int s, int e) {
		A = a;
		B = b;
		C = new float[a.size()][b.get(0).size()];
		start = s;
		end = e;
	}

	@Override
	public float[][] call() {
		for (int i = start; i < end; i++) {
			for (int k = 0; k < B.size(); k++) {
				for (int j = 0; j < B.get(0).size(); j++) {
					C[i][j] += A.get(i).get(k) * B.get(k).get(j);
				}
			}
		}
		return C;
	}
}