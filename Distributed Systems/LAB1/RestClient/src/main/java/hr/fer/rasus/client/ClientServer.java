package hr.fer.rasus.client;

import static hr.fer.rasus.utils.Utils.readCsv;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import static hr.fer.rasus.utils.Utils.*;
import hr.fer.rasus.dao.RawMeasurement;

public class ClientServer implements Runnable{
	public static String HOST_NAME = "localhost";
	private boolean runningFlag = true;
	private int port;
	private List<RawMeasurement> data;
	private Instant startTime;
	
	public ClientServer(int port) {
		this.port = port;
		this.data = readCsv(Sensor.DATA_PATH, RawMeasurement.class);
		this.startTime = Instant.now();
	}
	
	public RawMeasurement measure() {
		Instant currentTime = Instant.now();
		long ns = Duration.between(startTime, currentTime).toNanos();
		long sec = (long) ((long) ns / 1.0e9);
		int rb = (int)(sec % 100) + 2;
		return this.data.get(rb);
	}
	
	@Override
	public void run() {
		String threadName = Thread.currentThread().getName();
		System.out.println("SERVER THREAD: " + threadName);
		try(ServerSocket serverSocket = new ServerSocket(this.port)){
			System.out.format("Opened Server: (%s, %s)", serverSocket.getInetAddress(), serverSocket.getLocalPort());
			System.out.println(serverSocket);
			while(runningFlag) {
				System.out.println("WAITING: " + Thread.currentThread().getName());
				try( Socket clientSocket = serverSocket.accept(); //blocking
					BufferedReader inFromClient = new BufferedReader(
							                      new InputStreamReader(clientSocket.getInputStream()));
					PrintWriter outToClient = new PrintWriter(
							                  new OutputStreamWriter(clientSocket.getOutputStream()), true);)
				{
					//try-block
					String rcvString;
					while((rcvString = inFromClient.readLine()) != null) {
						rcvString = rcvString.trim();
						System.out.println("SERVER " + this.port + " RECIEVED " + rcvString);
						if(rcvString.equals("MEASURE")) {
							RawMeasurement raw = measure();
							System.out.println("server raw: " + raw);
							String json = toJson(raw);
							outToClient.println(json);
						}
						else if(rcvString.equals("CLOSE")){
							break;
						}
					}
				}
			}
			
		} catch (IOException e) {
			System.out.println("Cannot open socket with specified port number: " + port);
		}
	}
	
	public void setRunningFlag(boolean value) {
		runningFlag = value;
	}
	

}
