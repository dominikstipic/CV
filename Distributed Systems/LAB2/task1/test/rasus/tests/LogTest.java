package rasus.tests;


import org.junit.Test;

import hr.fer.rasus.utils.LogService;

public class LogTest {

	@Test
	public void logWriterTest() {
	LogService log = LogService.get();
	log.delete();
	log.print("Bok SVIMa\n");
	log.nl();
	log.print("ja programiram");
	}
}
