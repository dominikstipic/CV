package hr.fer.zemris.java.hw05.studentDB;

import static java.lang.Math.max;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import hr.fer.zemris.java.hw05.db.QueryFilter;
import hr.fer.zemris.java.hw05.db.QueryParser;
import hr.fer.zemris.java.hw05.db.StudentDatabase;
import hr.fer.zemris.java.hw05.db.StudentRecord;


/**
 * Omogucuje filritanje studentske baze podataka.
 * Putanja do tekstualnog zapisa baze se predaje preko 
 * komandne linije.
 * Prilikom pokretanja pokrece se jednostavna interakcija u kojoj korisnik 
 * postavlja upite.
 * Unosom rijeci "exit" izlazi se iz programa
 * @author Dominik Stipic
 *
 */
public class StudentDB {

	/**
	 * Automatski se pokrece prilikom pokretanja
	 * @param args 
	 */
	public static void main(String[] args) {
		@SuppressWarnings("unused")
		String str = "src/main/resources/database.txt";
		// tu se nalazi base.txt
		
		if(args.length != 1){
			System.out.println("Program couldnt find path to data base");
			System.exit(1);
		}
		String path = args[0];
		try {
			List<String> lines = DBloader(path);
			StudentDatabase dataBase = new StudentDatabase(lines);
			userInteraction(dataBase);
		} catch (IOException e) {
			System.out.println("Couldn't read data base lines");
			System.exit(1);
		}
	}
	
	/**
	 * metoda koja vrsi interakciju s korisnikom
	 * @param base baza podataka koja se koristi
	 */
	public static void userInteraction(StudentDatabase base) {
		Scanner input = new Scanner(System.in);
		
		while(true) {
			System.out.print(">  ");
			String query = input.nextLine().trim();
			if(query.equals("exit")) {
				System.out.println("Goodbye");
				break;
			}
			if(!query.matches("query .+")) {
				System.out.println("your input was incorrect -> didn't used query keyword");
				continue;
			}
			query = query.replaceAll("query", "").trim();
			QueryParser parser;
			List<StudentRecord> filtred = new LinkedList<>();
			try {
				parser = new QueryParser(query);
				if(parser.isDirectQuery()) {
					StudentRecord student = base.forJMBAG(parser.getQueriedJMBAG());
					System.out.println("Using index for record retrieval");
					filtred.add(student);
				}
				else {
					filtred = base.filter(new QueryFilter(parser.getQuery()));
				}
				printTable(filtred);
				filtred.clear();
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
			
		}
		input.close();
	}
	
	/**
	 * ispis tablica
	 * @param list lista studenta koji ce se ispisati
	 */
	private static void printTable(List<StudentRecord> list) {
		if(list.size() == 0) {
			System.out.println("Records selected: " + list.size());
			return;
		}
		int[] maximumLength = {0, 0 , 0, 0};
		for(StudentRecord record:list) {
			maximumLength[0] = max(maximumLength[0], record.getJMBAG().length());
			maximumLength[1] = max(maximumLength[1], record.getLastName().length());
			maximumLength[2] = max(maximumLength[2], record.getFirstName().length());
			maximumLength[3] = max(maximumLength[3], record.getFinalGrade().length());
		}
		printFrame(maximumLength);
		for(StudentRecord record:list) {
			StringBuilder builder = new StringBuilder();
			 builder.append(String.format("| %-" + maximumLength[0] + "s", record.getJMBAG())+" | ");
			 builder.append(String.format("%-" + maximumLength[1] + "s", record.getLastName())+" | ");
			 builder.append(String.format("%-" + maximumLength[2] + "s", record.getFirstName())+" | ");
			 builder.append(String.format("%-" + maximumLength[3] + "s", record.getFinalGrade())+" |");
			 System.out.println(builder.toString());
		}
		printFrame(maximumLength);
		System.out.println("Records selected: " + list.size());
	}
	
	/**
	 * crta okvir
	 * @param maximumLength velicina okvira
	 */
	private static void printFrame(int [] maximumLength) {
		for(int max : maximumLength) {
			System.out.print("+");
			for(int i = 0; i < max + 2;++i ) {
				System.out.print("=");
			}
		}
		System.out.print("+");
		System.out.println();
	}
	
	/**
	 * ucitava tekstaulnu datoteku u kojoj su zapisani podaci o studentima
	 * @param path putanja 
	 * @return lista linija iz tekstualne datoteke
	 * @throws IOException - ako nije moguce pronaci datoteku
	 */
	public static List<String>  DBloader(String path) throws IOException {
		return  Files.readAllLines(
				Paths.get(path),
				StandardCharsets.UTF_8
				);
	}

}
