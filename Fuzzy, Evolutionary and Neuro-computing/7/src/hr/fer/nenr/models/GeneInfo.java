package hr.fer.nenr.models;

import java.util.Arrays;

public class GeneInfo{
		public final double[] gene;
		public final int start;
		public final int end;
		public GeneInfo(double[] arr, int from, int end) {
			this.gene = arr;
			this.start = from;
			this.end = end;
		}
		
		public static GeneInfo extract(double[] geneSequence, int geneId, int[] geneLengths) {
			int targetLength = geneLengths[geneId];
			int start = 0;
			for(int i = 0; i < geneId; ++i) start += geneLengths[i];
			int to = start + targetLength;
			double[] neuronParam = Arrays.copyOfRange(geneSequence, start, to);
			return new GeneInfo(neuronParam, start, to);
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + end;
			result = prime * result + Arrays.hashCode(gene);
			result = prime * result + start;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			GeneInfo other = (GeneInfo) obj;
			if (end != other.end)
				return false;
			if (!Arrays.equals(gene, other.gene))
				return false;
			if (start != other.start)
				return false;
			return true;
		}

		@Override
		public String toString() {
			String str = Arrays.toString(gene) + "," + "start=" + start + "," + "end=" + end + "\n"; 
			return str;
		}
		
		
	}