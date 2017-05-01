import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class NegSampler {

	HashMap<Integer, Integer> u2idx;
	HashMap<Integer, Integer> idx2u;
	ArrayList<String> edges;
	double[] degreeDist;
	int numNodes;

	void readGraph(String fileName) throws IOException {
		edges = new ArrayList<>();
		u2idx = new HashMap<>();
		idx2u = new HashMap<>();

		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String line = "";
		int nodeCounter = 0;

		while ((line = br.readLine()) != null) {
			int a = Integer.parseInt(line.split(" ")[0]);
			int b = Integer.parseInt(line.split(" ")[1]);

			if (!u2idx.containsKey(a)) {
				u2idx.put(a, nodeCounter);
				idx2u.put(nodeCounter, a);
				nodeCounter += 1;
			}

			if (!u2idx.containsKey(b)) {
				u2idx.put(b, nodeCounter);
				idx2u.put(nodeCounter, b);
				nodeCounter += 1;
			}
			a = u2idx.get(a);
			b = u2idx.get(b);

			if (a < b) {
				edges.add(a + " " + b);
			} else
				edges.add(b + " " + a);
		}
		br.close();
		System.out.println(edges.size());
		Set<String> s = new HashSet<String>(edges);
		System.out.println(s.size());
		edges = new ArrayList<String>(s);
		numNodes = u2idx.size();
		System.out.println(edges.size());
	}

	double[] getDegreeDist() {
		double[] degDist = new double[numNodes];
		for (int i = 0; i < numNodes; i++)
			degDist[i] = 0;

		for (String e : edges) {
			int a = Integer.parseInt(e.split(" ")[0]);
			int b = Integer.parseInt(e.split(" ")[1]);
			degDist[a]++;
			degDist[b]++;
		}
		double sum = 0.0;
		for (int i = 0; i < numNodes; i++) {
			degDist[i] = Math.pow(degDist[i], 0.75);
			sum += degDist[i];
		}
		for (int i = 0; i < numNodes; i++) {
			degDist[i] /= sum;
		}
		return degDist;
	}

	void outputSamples(int numBatches, int batchSize, int negSamplesPerBatch, String fileName) throws IOException {
		Random random = new Random();
		List<Double> degDist = new ArrayList();
		for (int i = 0; i < numNodes; i++)
			degDist.add(degreeDist[i]);
		// System.out.println(degDist);
		AliasMethod am = new AliasMethod(degDist);
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		ArrayList<Integer> tempList = new ArrayList<>();
		for (int i = 0; i < numNodes; i++)
			tempList.add(i);

		for (int b = 0; b < numBatches; b++) {
			System.out.println("Batch " + b);
			int[] labels_batch = new int[batchSize * (1 + negSamplesPerBatch)];
			String[] edges_batch = new String[batchSize * (1 + negSamplesPerBatch)];
			Collections.shuffle(tempList);
			ArrayList<Integer> edges_list = new ArrayList<Integer>(tempList.subList(0, batchSize));
			int ctr = 0;
			for (int i = 0; i < batchSize; i++) {
				labels_batch[i] = 1;
				edges_batch[i] = edges.get(edges_list.get(i));
				ctr += 1;
			}

			for (int i = 0; i < batchSize; i++) {
				String edge_consider = edges_batch[i];
				String x = edge_consider.split(" ")[0];
				String y = edge_consider.split(" ")[1];

				for (int j = 0; j < negSamplesPerBatch; j++) {
					int neg_node = am.next();
					if (random.nextFloat() < 0.5)
						edges_batch[ctr] = x + " " + neg_node;
					else
						edges_batch[ctr] = y + " " + neg_node;
					
					labels_batch[ctr] = -1;
					ctr += 1;
				}
			}

			for (int i = 0; i < edges_batch.length; i++) {
				int s = idx2u.get(Integer.parseInt(edges_batch[i].split(" ")[0]));
				int t = idx2u.get(Integer.parseInt(edges_batch[i].split(" ")[1]));
				bw.write(s + "," + t + "," + labels_batch[i] + "\n");
				// bw.write(edges_batch[i] + " " + labels_batch[i] + "\n");
			}
			// bw.write("\n");
		}

		bw.close();
	}

	public static void main(String[] args) throws IOException {
		String fileName = args[0];
		int batchSize = Integer.parseInt(args[1]);
		int negSamplesPerBatch = Integer.parseInt(args[2]);
		int numBatches = Integer.parseInt(args[3]);
		// String fileName =
		// "/Users/aravind/Desktop/RASE/Datasets/Linkedin/network_edgelist.txt";
		// int batchSize = 100;
		// int negSamplesPerBatch = 5;
		// int numBatches = 5000;
		NegSampler ns = new NegSampler();
		ns.readGraph(fileName);
		ns.degreeDist = ns.getDegreeDist();
		System.out.println(ns.edges.size());

		ns.outputSamples(numBatches, batchSize, negSamplesPerBatch, "../../Batches_2nd.txt");
	}

}
