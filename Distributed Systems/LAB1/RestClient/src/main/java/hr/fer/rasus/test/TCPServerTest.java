package hr.fer.rasus.test;

import static hr.fer.rasus.utils.Utils.randomSleep;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Scanner;

import org.springframework.util.SocketUtils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import hr.fer.rasus.client.ClientServer;
import hr.fer.rasus.client.Sensor;
import hr.fer.rasus.dao.RawMeasurement;
import hr.fer.rasus.utils.Utils;

public class TCPServerTest {
	Sensor clientSensor = new Sensor();
	ClientServer server;
	Thread t;
	int SERVER_PORT;
	String SERVER_NAME = "localhost";
	Scanner s = new Scanner(System.in);

	public TCPServerTest() {
		SERVER_PORT = SocketUtils.findAvailableTcpPort();
		server = new ClientServer(SERVER_PORT);
		t = new Thread(server);
		t.start();
		mainJob();
	}

	public String communication(String msg) throws IOException {
		String rcvString = "";
		try (Socket clientSocket = new Socket(SERVER_NAME, SERVER_PORT);) {
			PrintWriter outToServer = new PrintWriter(
					                  new OutputStreamWriter(clientSocket.getOutputStream()), true);
			BufferedReader inFromServer = new BufferedReader(
					                      new InputStreamReader(clientSocket.getInputStream()));
			outToServer.println(msg);
			rcvString = inFromServer.readLine();
		}
		catch (Exception e) {
			System.out.println("ERROR WHILE CONNECTING");
		}
		return rcvString;
	}

	public void mainJob() {
		ObjectMapper mapper = new ObjectMapper();
		for (int i = 0; i < 3; ++i) {
			randomSleep(1000, 2000);
			try {
				CurrentTime ct = CurrentTime.generate();
				String time = mapper.writeValueAsString(ct);
				String fromServer = communication(time);
				System.out.println(fromServer);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.out.println("CLOSING");
		server.setRunningFlag(false);
		try {
			t.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		s.close();
	}

	public static void jsonTest() {
		CurrentTime t = CurrentTime.generate();
		ObjectMapper mapper = new ObjectMapper();
		try {
			String json = mapper.writeValueAsString(t);
			System.out.println(json);
			CurrentTime fromJson = mapper.readValue(json, CurrentTime.class);
			System.out.println(fromJson);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		//new TCPServerTest();
		//jsonTest();
		//Measurement raw = new Measurement("sensor1", "temp", null);
		
		//String json = Utils.toJson(raw);
		//System.out.println(json);
		
		RawMeasurement raw = new RawMeasurement(0.,0.,0.,null,0.4,0.);
		System.out.println(raw);
		String json = Utils.toJson(raw);
		System.out.println(json);
		RawMeasurement m = Utils.fromJson(json, RawMeasurement.class);
		System.out.println(m);
		
	}
	
}


