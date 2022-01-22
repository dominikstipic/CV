package hr.fer.rasus.dao;


public class Measurement {
	private Long id;
	private String username, parameter;
	private Float avarageValue;
	
	public Measurement(String username, String parameter, Float avarageValue) {
		this.username = username;
		this.parameter = parameter;
		this.avarageValue = avarageValue;
	}
	
	public Measurement() {}
	
	public String getUsername() {
		return username;
	}
	public void setUsername(String username) {
		this.username = username;
	}
	public String getParameter() {
		return parameter;
	}
	public void setParameter(String parameter) {
		this.parameter = parameter;
	}
	public Float getAvarageValue() {
		return avarageValue;
	}
	public void setAvarageValue(Float avarageValue) {
		this.avarageValue = avarageValue;
	}
	public Long getId() {
		return id;
	}
	public void setId(Long id) {
		this.id = id;
	}
	@Override
	public String toString() {
		return "Measurement [id=" + id + ", username=" + username + ", parameter=" + parameter + ", avarageValue="
				+ avarageValue + "]";
	}
	
	
	
	
	
	
}
