package hr.fer.rasus.dao;

import java.util.ArrayList;
import java.util.List;

public class RawMeasurement {
	private Double temperature, pressure, humidity, co, no2, so2;

	public RawMeasurement(double temperature, double pressure, double humidity, double co, double no2, double so2) {
		this.temperature = temperature;
		this.pressure = pressure;
		this.humidity = humidity;
		this.co = co;
		this.no2 = no2;
		this.so2 = so2;
	}
	
	public RawMeasurement() {}
	
	public double getTemperature() {
		return temperature;
	}

	public void setTemperature(double temperature) {
		this.temperature = temperature;
	}

	public double getPressure() {
		return pressure;
	}

	public void setPressure(double pressure) {
		this.pressure = pressure;
	}

	public double getHumidity() {
		return humidity;
	}

	public void setHumidity(double humidity) {
		this.humidity = humidity;
	}

	public double getCo() {
		return co;
	}

	public void setCo(double co) {
		this.co = co;
	}

	public double getNo2() {
		return no2;
	}

	public void setNo2(double no2) {
		this.no2 = no2;
	}

	public double getSo2() {
		return so2;
	}

	public void setSo2(double so2) {
		this.so2 = so2;
	}

	public List<String> getParams(){
		return List.of("temperature", "pressure", "humidity", "co", "no2", "so2");
	}
	
	public List<Double> getValues(){
		List<Double> list = new ArrayList<>();
		list.add(temperature);
		list.add(pressure);
		list.add(humidity);
		list.add(co);
		list.add(no2);
		list.add(so2);
		return list;
	}
	
	@Override
	public String toString() {
		return "RawMeasurement [temperature=" + temperature + ", pressure=" + pressure + ", humidity=" + humidity
				+ ", co=" + co + ", no2=" + no2 + ", so2=" + so2 + "]";
	}
	
	
}
