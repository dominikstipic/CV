package hr.fer.rasus.dao;

import java.util.ArrayList;
import java.util.List;

import hr.fer.rasus.time.TimeLabel;

public class RawMeasurement {
	private Double temperature, pressure, humidity, co, no2, so2;
	private TimeLabel scalar;
	private TimeLabel vector;
	private Long systemTime;

	public RawMeasurement(Double temperature, Double pressure, Double humidity, Double co, Double no2, Double so2) {
		this.temperature = temperature;
		this.pressure = pressure;
		this.humidity = humidity;
		this.co = co;
		this.no2 = no2;
		this.so2 = so2;
	}
	
	public TimeLabel getScalar() {
		return scalar;
	}

	public void setScalar(TimeLabel scalar) {
		this.scalar = scalar;
	}

	public TimeLabel getVector() {
		return vector;
	}

	public void setVector(TimeLabel vector) {
		this.vector = vector;
	}

	public Long getSystemTime() {
		return systemTime;
	}

	public void setSystemTime(Long systemTime) {
		this.systemTime = systemTime;
	}

	public RawMeasurement(List<Double> values) {
		this.temperature = values.get(0);
		this.pressure = values.get(1);
		this.humidity = values.get(2);
		this.co = values.get(3);
		this.no2 = values.get(4);
		this.so2 = values.get(5);
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

	public static List<String> params(){
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
	
	public void updateTime(TimeLabel scalar, TimeLabel vector) {
		setVector(vector);
		setScalar(scalar);
	}
	
	@Override
	public String toString() {
		String str = "[temperature=" + temperature + ", pressure=" + pressure + ", humidity=" + humidity
				+ ", co=" + co + ", no2=" + no2 + ", so2=" + so2 + "]";
		String scalarStr = "";
		String vectorStr = "";
		
		if(scalar != null)
			scalarStr = "SCALAR: " + scalar.getValues();
		if(vector != null)
			vectorStr = "VECTOR: " + vector.getValues();
		String string = String.format("%s\t%s\t%s", str, scalarStr, vectorStr);
		return string;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((co == null) ? 0 : co.hashCode());
		result = prime * result + ((humidity == null) ? 0 : humidity.hashCode());
		result = prime * result + ((no2 == null) ? 0 : no2.hashCode());
		result = prime * result + ((pressure == null) ? 0 : pressure.hashCode());
		result = prime * result + ((scalar == null) ? 0 : scalar.hashCode());
		result = prime * result + ((so2 == null) ? 0 : so2.hashCode());
		result = prime * result + ((temperature == null) ? 0 : temperature.hashCode());
		result = prime * result + ((vector == null) ? 0 : vector.hashCode());
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
		RawMeasurement other = (RawMeasurement) obj;
		if (co == null) {
			if (other.co != null)
				return false;
		} else if (!co.equals(other.co))
			return false;
		if (humidity == null) {
			if (other.humidity != null)
				return false;
		} else if (!humidity.equals(other.humidity))
			return false;
		if (no2 == null) {
			if (other.no2 != null)
				return false;
		} else if (!no2.equals(other.no2))
			return false;
		if (pressure == null) {
			if (other.pressure != null)
				return false;
		} else if (!pressure.equals(other.pressure))
			return false;
		if (scalar == null) {
			if (other.scalar != null)
				return false;
		} else if (!scalar.equals(other.scalar))
			return false;
		if (so2 == null) {
			if (other.so2 != null)
				return false;
		} else if (!so2.equals(other.so2))
			return false;
		if (temperature == null) {
			if (other.temperature != null)
				return false;
		} else if (!temperature.equals(other.temperature))
			return false;
		if (vector == null) {
			if (other.vector != null)
				return false;
		} else if (!vector.equals(other.vector))
			return false;
		return true;
	}

	
	
	
	
	
	
}
