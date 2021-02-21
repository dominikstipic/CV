package nenr.fuzzy.system;

import nenr.fuzzy.database.AkcelDatabaseFactory;
import nenr.fuzzy.database.FuzzyDatabase;
import nenr.fuzzy.database.KormiloDatabaseFactory;

public class FuzzySystemFactory {
	
	public static IFuzzySystem akcelFuzzy() {
		FuzzyDatabase db = new AkcelDatabaseFactory().createDatabase();
		FuzzySystem fuzzySystem = new FuzzySystem(db);
		return fuzzySystem;
	}
	
	public static IFuzzySystem kormiloFuzzy() {
		FuzzyDatabase db = new KormiloDatabaseFactory().createDatabase();
		FuzzySystem fuzzySystem = new FuzzySystem(db);
		return fuzzySystem;
	}
}
