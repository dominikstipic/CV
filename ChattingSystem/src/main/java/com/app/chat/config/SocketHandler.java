package com.app.chat.config;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;


@Component
public class SocketHandler extends TextWebSocketHandler {
	
	private List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();
	private List<String> usernames = new LinkedList<String>();

	@Override
	public void handleTextMessage(WebSocketSession session, TextMessage message) throws InterruptedException, IOException {
		String value = message.getPayload();
		if(value.startsWith("#username")) {
			String name = value.split(":")[1];
			usernames.add(name);
		}
		else {
			String username = getUserName(session);
			for(WebSocketSession webSocketSession : sessions) {
				webSocketSession.sendMessage(new TextMessage(username + ":" + value));
			}
		}
	}

	@Override
	public void afterConnectionEstablished(WebSocketSession session) throws Exception {
		System.out.println("new Connection");
		sessions.add(session);
	}

	@Override
	public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
		System.out.println("Session Disconnection");
		String username = getUserName(session);
		usernames.remove(username);
		sessions.remove(session);
		
		for(WebSocketSession webSocketSession : sessions) {
			webSocketSession.sendMessage(new TextMessage(username + " leaved"));
		}
	}
	
	private String getUserName(WebSocketSession session) {
		String username = "User";
		for(int i = 0; i < sessions.size(); ++i) {
			if(session.equals(sessions.get(i))) {
				username = usernames.get(i);
				break;
			}
		}
		return username;
	}
	
	
}