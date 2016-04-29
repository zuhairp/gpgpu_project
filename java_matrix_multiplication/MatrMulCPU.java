/* Modified from https://martin-thoma.com/part-iii-matrix-multiplication-on-multiple-cores-in-python-java-and-c/ */

import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MatrMulCPU {
	

	public static int MAX_N = 8192;
	public static int MAX_P = 8192;
	public static int MAX_M = 256;

	static void printMatrix(float[][] matrix) {
		for (float[] line : matrix) {
			int i = 0;
			StringBuilder sb = new StringBuilder(matrix.length);
			for (float number : line) {
				if (i != 0) {
					sb.append("\t");
				} else {
					i++;
				}
				sb.append(number);
			}
			System.out.println(sb.toString());
		}
	}

	public static float[][] parallelMult(ArrayList<ArrayList<Float>> A,
			ArrayList<ArrayList<Float>> B, int threadNumber) {
		float[][] C = new float[A.size()][B.get(0).size()];
		ExecutorService executor = Executors.newFixedThreadPool(threadNumber);
		List<Future<float[][]>> list = new ArrayList<Future<float[][]>>();

		int part = A.size() / threadNumber;
		if (part < 1) {
			part = 1;
		}
		for (int i = 0; i < A.size(); i += part) {
			Callable<float[][]> worker = new LineMultiplier(A, B, i, i+part);
			Future<float[][]> submit = executor.submit(worker);
			list.add(submit);
		}

		// now retrieve the result
		int start = 0;
		float CF[][];
		for (Future<float[][]> future : list) {
			try {
				CF = future.get();
				for (int i=start; i < start+part; i += 1) {
					C[i] = CF[i];
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			start+=part;
		}
		executor.shutdown();

		return C;
	}

	public static void main(String[] args) {
		int cores = Runtime.getRuntime().availableProcessors();
		System.err.println("Number of cores:\t" + cores);
		
		ArrayList<ArrayList<Object>> data = new ArrayList<ArrayList<Object>>();
		Random rand = new Random();
		
		for (int n = 1; n <= MAX_N; n*=2) {
			for (int p = 1; p <= MAX_P; p*=2) {
				for (int m = 1; m <= MAX_M; m*=2) {
					ArrayList<ArrayList<Float>> A = new ArrayList<ArrayList<Float>>();
					ArrayList<ArrayList<Float>> B = new ArrayList<ArrayList<Float>>();
					
					for (int i = 0; i < n; i++) {   
						ArrayList<Float> temp = new ArrayList<Float>();
				        for (int j = 0; j < m; j++) {
				           temp.add(rand.nextFloat());
				        }
				        A.add(temp);
				    }
					for (int i = 0; i < m; i++) {    
						ArrayList<Float> temp = new ArrayList<Float>();
				        for (int j = 0; j < p; j++) {
				        	temp.add(rand.nextFloat());
				        }
				        B.add(temp);
				    }
					
					long startTime = System.nanoTime();
					float[][] C = parallelMult(A, B, cores);
					long endTime = System.nanoTime();
					String delta_secs = new BigDecimal((endTime - startTime)/(10e9)).toPlainString();
					ArrayList<Object> temp_data = new ArrayList<Object>();
					temp_data.add(n*p*m);
					temp_data.add(delta_secs);
					data.add(temp_data);
					System.out.println("n*p*m=" + n*p*m + " n=" + n + " m=" + m + " p=" + p);
					System.gc();
				}
			}
		}
		
		try
		{
		    FileWriter writer = new FileWriter("./data_cpu.csv");
			 
		    for (ArrayList<Object> arr : data) {
		    	for (Object obj: arr) {
		    		writer.append(obj.toString() + ", ");
		    	}
		    	writer.append("\n");
		    }
				
		    writer.flush();
		    writer.close();
		}
		catch(IOException e)
		{
		     e.printStackTrace();
		} 
		System.out.println("done");
		
	}
}
