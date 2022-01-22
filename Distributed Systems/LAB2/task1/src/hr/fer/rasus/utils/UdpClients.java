package hr.fer.rasus.utils;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

public class UdpClients {
	public static void sendPacket(byte[] buff,  int port) {
		try(DatagramSocket clientSocket = new DatagramSocket()) {
			InetAddress targetIp = InetAddress.getByName("localhost");
			DatagramPacket p = new DatagramPacket(buff, buff.length, targetIp, port);
			clientSocket.send(p);
		} catch (Exception e) {
			//e.printStackTrace();
		}
	}
}
