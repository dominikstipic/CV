package hr.fer.zemris.java.hw05.db;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import hr.fer.zemris.java.hw05.collections.SimpleHashtable;

/**
 * Predstavlja bazu podataka koja sadrzi studente modelirane uz pomoc
 * <code>StudentRecord</code>.Student je definiran s sljedecim svojsvima:
 * jmbag,ime,prezime,ocjena.
 * Baza omogucava fitriranje, te dohvacanje pojedinog studenta u O(1) slozenosti.
 * @author Dominik Stipic
 *
 */
public class StudentDatabase {
	/**
	 * Lista studenta 
	 */
	private List<StudentRecord> recordList;
	/**
	 * Tablica rasprsenog adresiranja koja sadrzi studente
	 */
	private SimpleHashtable<String, StudentRecord> dataBase;

	/**
	 * Konstuktor koji stvara bazu uz pomoc tekstualne datoteke ispunjene svim potrebnim podacima
	 * @param lines lista String reprezentacija studenta
	 */
	public StudentDatabase(List<String> lines) {
		Objects.requireNonNull(lines, "data base lines mustn't be null");
		recordList = new LinkedList<>();
		for(String line:lines) {
			String parts[] = line.split("[ ]+|\\t+");
			if(parts.length == 4 ) {
				recordList.add(new StudentRecord(parts[0], parts[2], parts[1], parts[3]));
			}
			else {
				String lastName = parts[1] + " " + parts[2];
				recordList.add(new StudentRecord(parts[0], parts[3], lastName, parts[4]));
			}
		}
		
		dataBase = new SimpleHashtable<>(recordList.size());
		for(StudentRecord r:recordList) {
			dataBase.put(r.getJMBAG(), r);
		}
	}
	
	/**
	 * Dohvaca studenta uz pomoc njegovog identifikatora JMBAG u O(1) slozenosti
	 * @param JMBAG idetifikator
	 * @return student
	 * @throws NullPointerException ako je kljuc null
	 */
	public StudentRecord forJMBAG (String JMBAG) {
		Objects.requireNonNull(JMBAG, "JMBAG cannot be null");
		return dataBase.get(JMBAG);
	}
	
	/**
	 * Filtrira bazu podataka vracajuci listu studenata koji zadovolajvaju dana pravila 
	 * @param filter objekt koji zna pod koji uvjetima filtrirati 
	 * @return Listu fitriranih studenata
	 */
	public List<StudentRecord> filter (IFilter filter){
		List<StudentRecord> filtered = new LinkedList<>();
		
		for(StudentRecord r:recordList) {
			if(filter.accepts(r)) {
				filtered.add(r);
			}
		}
		return filtered;
	}

	/**
	 * getter
	 * @return lista svih studenata u bazi
	 */
	public List<StudentRecord> getRecordList() {
		return recordList;
	}
	
	
}
