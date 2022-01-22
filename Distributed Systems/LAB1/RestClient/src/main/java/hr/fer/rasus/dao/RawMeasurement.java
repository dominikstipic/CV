package hr.fer.rasus.dao;

import java.util.ArrayList;
import java.util.List;

public class RawMeasurement {
	private Double temperature, pressure, humidity, co, no2, so2;

	public RawMeasurement(Double temperature, Double pressure, Double humidity, Double co, Double no2, Double so2) {
		this.temperature = temperature;
		this.pressure = pressure;
		this.humidity = humidity;
		this.co = co;
		this.no2 = no2;
		this.so2 = so2;
	}
	
	public RawMeasurement() {}
	
	public Double getTemperature() {
		return temperature;
	}

	public void setTemperature(Double temperature) {
		this.temperature = temperature;
	}

	public Double getPressure() {
		return pressure;
	}

	public void setPressure(Double pressure) {
		this.pressure = pressure;
	}

	public Double getHumidity() {
		return humidity;
	}

	public void setHumidity(Double humidity) {
		this.humidity = humidity;
	}

	public Double getCo() {
		return co;
	}

	public void setCo(Double co) {
		this.co = co;
	}

	public Double getNo2() {
		return no2;
	}

	public void setNo2(Double no2) {
		this.no2 = no2;
	}

	public Double getSo2() {
		return so2;
	}

	public void setSo2(Double so2) {
		this.so2 = so2;
	}

	public List<String> params(){
		return List.of("temperature", "pressure", "humidity", "co", "no2", "so2");
	}
	
	public List<Double> values(){
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
