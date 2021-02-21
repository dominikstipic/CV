package hr.fer.nenr.utils;

public enum Label {
	ALPHA("ALPHA"), 
	BETA("BETA"), 
	GAMMA("GAMMA"),
	DELTA("DELTA"), 
	ETHA("ETHA");
	
	public final String label;
	public final int value;
	
	private Label(String label) {
		this.label = label;
		switch(label) {
		case "ALPHA":
			value = 0;
			break;
		case "BETA":
			value = 1;
			break;
		case "GAMMA":
			value = 2;
			break;
		case "DELTA":
			value = 3;
			break;
		case "ETHA":
			value = 4;
			break;
		default:
			value=0;
		}
	}
	
	public static Label getLabel(String label) {
		Label val;
		switch(label) {
		case "ALPHA":
			val = ALPHA;
			break;
		case "BETA":
			val = BETA;
			break;
		case "GAMMA":
			val = GAMMA;
			break;
		case "DELTA":
			val = DELTA;
			break;
		case "ETHA":
			val = ETHA;
			break;
		default:
			val = ALPHA;
		}
		return val;
	}
	
	public static Label fromValue(int v) {
		Label val;
		switch(v) {
		case 0:
			val = ALPHA;
			break;
		case 1:
			val = BETA;
			break;
		case 2:
			val = GAMMA;
			break;
		case 3:
			val = DELTA;
			break;
		case 4:
			val = ETHA;
			break;
		default:
			throw new IllegalArgumentException("Label doesn't exist");
		}
		return val;
	}
}
