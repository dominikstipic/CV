package hr.fer.rasus.dao;

public class UserAddress {
	private String username, ipAddress; 
	int port;

	public UserAddress(String username, String ipAddress, int port) {
		this.username = username;
		this.ipAddress = ipAddress;
		this.port = port;
	}

	public UserAddress() {}

	public String getIpAddress() {
		return ipAddress;
	}

	public void setIpAddress(String ipAddress) {
		this.ipAddress = ipAddress;
	}

	public int getPort() {
		return port;
	}

	public void setPort(int port) {
		this.port = port;
	}

	public String getUsername() {
		return username;
	}

	public void setUsername(String username) {
		this.username = username;
	}

	@Override
	public String toString() {
		return "UserAddress [username=" + username + ", ipAddress=" + ipAddress + ", port=" + port + "]";
	}
}
