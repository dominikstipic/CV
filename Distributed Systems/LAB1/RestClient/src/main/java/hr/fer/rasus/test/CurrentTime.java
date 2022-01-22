package hr.fer.rasus.test;

import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;

public class CurrentTime{
	private int sec;
	private int hour;
	private int minute;
	private int day;
	private String month;
	private int year;

	public CurrentTime(int sec, int hour, int minute, int day, String month, int year) {
		this.sec = sec;
		this.hour = hour;
		this.minute = minute;
		this.day = day;
		this.month = month;
		this.year = year;
	}
	
	public CurrentTime() {}
	
	public int getHour() {
		return hour;
	}

	public void setHour(int hour) {
		this.hour = hour;
	}

	public int getMinute() {
		return minute;
	}

	public void setMinute(int minute) {
		this.minute = minute;
	}

	public int getDay() {
		return day;
	}

	public void setDay(int day) {
		this.day = day;
	}

	public String getMonth() {
		return month;
	}

	public void setMonth(String month) {
		this.month = month;
	}

	public int getYear() {
		return year;
	}

	public void setYear(int year) {
		this.year = year;
	}
	
	public int getSec() {
		return sec;
	}

	public void setSec(int sec) {
		this.sec = sec;
	}

	@Override
	public String toString() {
		return "CurrentTime [sec=" + sec + ", hour=" + hour + ", minute=" + minute + ", day=" + day + ", month=" + month
				+ ", year=" + year + "]";
	}

	public static CurrentTime generate() {
		ZonedDateTime time = Instant.now().atZone(ZoneId.systemDefault());
		CurrentTime t = new CurrentTime(time.getSecond(),
				                        time.getHour(), 
				                        time.getMinute(), 
				                        time.getDayOfMonth(), 
				                        time.getMonth().toString(), 
				                        time.getYear());
		return t;
	}
}
