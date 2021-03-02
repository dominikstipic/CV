package hr.fer.zemris.java.tests.dataBase;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.db.ComparisonOperators;
import hr.fer.zemris.java.hw05.db.ConditionalExpression;
import hr.fer.zemris.java.hw05.db.FieldValueGetters;
import hr.fer.zemris.java.hw05.db.StudentRecord;

public class ConditionalExpressionTest {
	
	@Test
	public void forCondExpression() {
		ConditionalExpression expr = new ConditionalExpression(
				FieldValueGetters.LAST_NAME,
				ComparisonOperators.LIKE,
				"Bos*"
				);
		
				StudentRecord record = new StudentRecord("0", "Marko", "Bosnic", "***");
				
				boolean recordSatisfies = expr.getComparisonOperator().satisfied(
				expr.getFieldGetter().get(record), 
				expr.getStringLiteral() 
				);
				
				Assert.assertEquals(true ,recordSatisfies);
				
				StudentRecord record1 = new StudentRecord("0", "Marko", "Markic", "***");
				
				boolean value = expr.getComparisonOperator().satisfied(expr.getFieldGetter().get(record1), expr.getStringLiteral());
				
				Assert.assertEquals(false ,value);
	}
}
