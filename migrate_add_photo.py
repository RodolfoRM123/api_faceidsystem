"""
Script para actualizar la base de datos con el nuevo campo photo_path
"""
import psycopg2

def migrate():
    # Conectar directamente a PostgreSQL
    conn = psycopg2.connect(
        host="62.72.1.252",
        port=5433,
        database="segundacuna",
        user="master",
        password="123456"
    )
    
    try:
        cursor = conn.cursor()
        
        # Verificar si la columna ya existe
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='photo_path';
        """)
        
        if cursor.fetchone():
            print("La columna photo_path ya existe")
        else:
            # Agregar columna photo_path
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN photo_path VARCHAR;
            """)
            conn.commit()
            print("Migracion completada: Campo photo_path agregado a la tabla users")
        
        cursor.close()
    except Exception as e:
        print(f"Error en migracion: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()

