package hr.fer.rasus.utils;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

import hr.fer.rasus.subcomponents.SimpleSimulatedDatagramSocket;

public class UdpClients {
	public static final double lossRate = 0.2; 
	public static final int averageDelay = 1000;
	
	private static void simpleClient(byte[] buff,  int port) {
		try(DatagramSocket clientSocket = new DatagramSocket()) {
			InetAddress targetIp = InetAddress.getByName("localhost");
			DatagramPacket p = new DatagramPacket(buff, buff.length, targetIp, port);
			clientSocket.send(p);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void simulatedClient(byte[] buff,  int port) {
		//new SimulatedDatagramSocket(lossRate, 1, averageDelay, 50);
		//new SimpleSimulatedDatagramSocket(lossRate, averageDelay)
		try(DatagramSocket clientSocket = new SimpleSimulatedDatagramSocket(lossRate, averageDelay)) {
			InetAddress targetIp = InetAddress.getByName("localhost");
			DatagramPacket p = new DatagramPacket(buff, buff.length, targetIp, port);
			clientSocket.send(p);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void sendPacket(byte[] buff,  int port, boolean isSimulated) {
		if(!isSimulated) simpleClient(buff, port);
		else simulatedClient(buff, port);
	}
	
}
