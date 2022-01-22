package hr.fer.rasus.utils;
import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.dao.Tuple2Types;

public class Protocol {
	public static final String END = "END";
	public static final String SHARE = "SHARE";
	public static final String REPLY = "REPLY";
	public static final String TRIGGER_HEADER = "TRIGGER";
	
	
	public static boolean isEnd(String message) {
		return message.equals(END);
	}
	
	///////
	
	public static boolean isTrigger(String message) {
		return message.startsWith(TRIGGER_HEADER);
	}
	
	public static String triggerMessage(Long periodStart, Long periodEnd) {
		String str = String.format("TRIGGER %s %s", periodStart.toString(), periodEnd.toString());
		return str;
	}
	
	public static Long[] repackTrigger(String message) {
		String[] arr = message.split(" ");
		Long t1 = Long.valueOf(arr[1]);
		Long t2 = Long.valueOf(arr[2]);
		return new Long[]{t1,t2};
	}
	
	///////
	
	public static String shareMessage(String origin, String json) {
		String message = String.format("SHARE %s\nDATA %s", origin, json);
		return message;
	}
	
	public static Tuple2Types<String, RawMeasurement> repackShare(String message) {
		String newline = System.lineSeparator();
		String[] arr = message.split(newline);
		String name = arr[0].split(" ")[1];
		String json = arr[1].split(" ")[1];
		RawMeasurement data = Utils.fromJSON(json, RawMeasurement.class);
		Tuple2Types<String, RawMeasurement> tuple = new Tuple2Types<>();
		tuple.s = name;
		tuple.t = data;
		return tuple;
	}
	
	public static boolean isShare(String message) {
		return message.startsWith(Protocol.SHARE);
	}
	
	////
	
	public static String replyMessage(String name, int code) {
		String str = String.format("REPLY %s %s", name, code);
		return str;
	}
	
	public static boolean isReply(String message) {
		return message.startsWith(REPLY);
	}
	
	public static Tuple2Types<String, Integer> repackReply(String message) {
		String[] arr = message.split(" ");
		String name = arr[1]; String num = arr[2];
		int code = Integer.valueOf(num);
		Tuple2Types<String, Integer> tuple = new Tuple2Types<>();
		tuple.s = name;
		tuple.t = code;
		return tuple;
	}
}
