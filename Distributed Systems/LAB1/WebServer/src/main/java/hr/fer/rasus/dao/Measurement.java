package hr.fer.rasus.dao;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

import com.sun.istack.Nullable;

@Entity
public class Measurement {
	@Id
	@GeneratedValue(strategy = GenerationType.AUTO)
	private Long id;
	private String username, parameter;
	@Nullable
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
