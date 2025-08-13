from flask import jsonify
import mysql
import mysql.connector
from mysql.connector import Error

class DBHelper:
    DB_CONFIG = {
        'host': 'db.dev.erp.mdi',
        'user': 'phpdev',
        'password': 'phpdev',
        'database': 'phpdevs'
    }

    def get_connection(self):
        try:
            return mysql.connector.connect(**self.DB_CONFIG)
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None



    def GetSavedContacts(self, owner_mobile):
        conn = self.get_connection()
        if conn is None:
            return {}

        try:
            cursor = conn.cursor()
            query = """
                SELECT c.contact_mobile, COALESCE(c.nickname, u.username) AS display_name
                FROM contacts_sign_chat c
                JOIN users_sign_chat u ON c.contact_mobile = u.mobile
                WHERE c.owner_mobile = %s
                AND u.is_active = 1
            """
            cursor.execute(query, (owner_mobile,))
            results = cursor.fetchall()

            contacts_dict = {row[0]: row[1] for row in results}  # mobile: display_name
            return contacts_dict

        except Exception as e:
            print(f"Error fetching saved contacts: {e}")
            return {}

        finally:
            cursor.close()
            conn.close()

    def DeleteMessage(self, message_id):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            query = "DELETE FROM messages_sign_chat WHERE id = %s"
            cursor.execute(query, (message_id,))
            conn.commit()
            return True

        except Exception as e:
            print(f"Error deleting message: {e}")
            return False

        finally:
            cursor.close()
            conn.close()

    def StoreFeedback(self, message_id, feedback_text):
            conn = self.get_connection()
            if conn is None:
             return False
            try:
                cursor = conn.cursor()
                print(f"{message_id}  meassage id  and  {feedback_text} feedback  database")
                query = """
                    UPDATE messages_sign_chat
                    SET feedback = %s
                    WHERE id = %s
                """
                cursor.execute("UPDATE messages_sign_chat SET feedback = %s WHERE id = %s", (feedback_text, message_id))

                conn.commit()

                return {"success": True, "message": "Feedback submitted successfully!"}

            except Exception as e:
                print(f"Error storing feedback: {e}")
                return False

            finally:
                cursor.close()
                conn.close()
    def GetSavedContacts1(self, owner_mobile):
        conn = self.get_connection()
        if conn is None:
            return []

        try:
            cursor = conn.cursor()
            query = """
                SELECT
                    c.contact_mobile AS mobile,
                    c.nickname,
                    (
                        SELECT COUNT(*)
                        FROM messages_sign_chat
                        WHERE sender_mobile = c.contact_mobile
                        AND receiver_mobile = %s
                        AND seen = 0
                    ) AS unread
                FROM contacts_sign_chat c
                WHERE c.owner_mobile = %s
            """
            cursor.execute(query, (owner_mobile, owner_mobile))
            results = cursor.fetchall()

            return [{"mobile": row[0], "nickname": row[1], "unread": row[2]} for row in results]

        except Exception as e:
            print(f"Error fetching saved contacts: {e}")
            return []

        finally:
            cursor.close()
            conn.close()


    def SaveContact(self, owner_mobile, contact_mobile, nickname):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            query = """
                INSERT IGNORE INTO contacts_sign_chat (owner_mobile, contact_mobile, nickname)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (owner_mobile, contact_mobile, nickname))
            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving contact: {e}")
            return False

        finally:
            cursor.close()
            conn.close()


    def SaveChatMessage(self, sender_mobile, receiver_mobile, message_content):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO messages_sign_chat
                (sender_mobile, receiver_mobile, message_content, seen, status)
                VALUES (%s, %s, %s, 0, 'sent')
            """
            cursor.execute(query, (sender_mobile, receiver_mobile, message_content))
            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False

        finally:
            cursor.close()
            conn.close()

    def SearchContact(self, owner_mobile, search_keyword):
        conn = self.get_connection()
        if conn is None:
            return {"status": "error"}

        try:
            cursor = conn.cursor()
            like_search = f"{search_keyword}%"

            contact_list = []

            # Step 1: Search Saved Contacts (by name or number)
            cursor.execute("""
                SELECT nickname, contact_mobile FROM contacts_sign_chat
                WHERE owner_mobile = %s AND (contact_mobile LIKE %s OR nickname LIKE %s)
            """, (owner_mobile, like_search, like_search))

            saved_contacts = cursor.fetchall()
            for contact in saved_contacts:
                contact_list.append({
                    "name": contact[0],
                    "mobile": contact[1],
                    "status": "saved"
                })

            # Step 2: Search Registered Users Not Already Saved (by number or name)
            cursor.execute("""
                SELECT username, mobile FROM users_sign_chat
                WHERE is_active = 1 AND mobile !=%s AND
                    (mobile LIKE %s) AND
                    mobile NOT IN (
                        SELECT contact_mobile FROM contacts_sign_chat WHERE owner_mobile = %s
                    )
            """, (owner_mobile, like_search,  owner_mobile))

            registered_users = cursor.fetchall()
            for user in registered_users:
                contact_list.append({
                    "name": user[0],
                    "mobile": user[1],
                    "status": "registered"
                })

            if contact_list:
                return {"status": "multiple", "contacts": contact_list}
            else:
                return {"status": "not_registered"}

        except Exception as e:
            print(f"Error searching contact: {e}")
            return {"status": "error"}
        finally:
            try:
                if cursor:
                    cursor.close()
            except:
                pass
            conn.close()



    def GetChatMessages(self, user_mobile, contact_mobile):
        conn = self.get_connection()
        if conn is None:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT
                    id AS message_id,
                    sender_mobile,
                    receiver_mobile,
                    message_content,
                    seen,
                    DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS created_at
                FROM messages_sign_chat
                WHERE
                    (sender_mobile = %s AND receiver_mobile = %s)
                    OR (sender_mobile = %s AND receiver_mobile = %s)
                ORDER BY created_at ASC
            """
            cursor.execute(query, (user_mobile, contact_mobile, contact_mobile, user_mobile))
            results = cursor.fetchall()
            return results

        except Exception as e:
            print(f"Error fetching chat messages: {e}")
            return []

        finally:
            cursor.close()
            conn.close()



    def get_user_by_mobile(self, mobile):
        conn = self.get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users_sign_chat WHERE mobile = %s", (mobile,))
            return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching user by mobile: {e}")
            return None
        finally:
            conn.close()


    def user_exists(self, mobile):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users_sign_chat WHERE mobile = %s", (mobile,))
            return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking if user exists: {e}")
            return False
        finally:
            conn.close()


    def mark_messages_as_seen(self, sender_mobile, receiver_mobile):
        conn = self.get_connection()  # Tumhara db connection logic
        cursor = conn.cursor()

        sql = """
            UPDATE messages_sign_chat
            SET seen = 1
            WHERE sender_mobile = %s AND receiver_mobile = %s AND seen = 0
        """
        cursor.execute(sql, (sender_mobile, receiver_mobile))
        conn.commit()
        cursor.close()
        conn.close()

    def register_user(self, mobile, username, email, gender, password_hash, profile_picture):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO users_sign_chat
                (mobile, username, email, gender, password_hash, profile_picture)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (mobile, username, email, gender, password_hash, profile_picture))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error registering user: {e}")
            return False
        finally:
            conn.close()

    def get_profile_picture(self, mobile):
        conn = self.get_connection()
        if conn is None:
            return '/static/uploads/blank.png'

        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT profile_picture FROM users_sign_chat WHERE mobile = %s"
            cursor.execute(query, (mobile,))
            row = cursor.fetchone()

            if row and row['profile_picture']:
                return row['profile_picture']
            else:
                return '/static/uploads/blank.png'

        except Exception as e:
            print(f"Error fetching profile picture for {mobile}: {e}")
            return '/static/uploads/blank.png'

        finally:
            cursor.close()
            conn.close()

    def update_contact_name(self, user_mobile, contact_mobile, new_name):
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()
            query = """
                UPDATE contacts_sign_chat
                SET nickname = %s
                WHERE owner_mobile = %s AND contact_mobile = %s
            """
            cursor.execute(query, (new_name, user_mobile, contact_mobile))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating contact name: {e}")
            return False
        finally:
            if conn:
                conn.close()


        # Add new method to DBHelper class
    def update_user_profile(self, mobile, username, email, gender, profile_picture):
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            query = """
                UPDATE users_sign_chat
                SET username = %s,
                    email = %s,
                    gender = %s,
                    profile_picture = %s
                WHERE mobile = %s
            """
            cursor.execute(query, (username, email, gender, profile_picture, mobile))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False
        finally:
            conn.close()

    def delete_contact(self, user_mobile, contact_mobile):
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()
            query = """
                DELETE FROM contacts_sign_chat
                WHERE owner_mobile = %s AND contact_mobile = %s
            """
            cursor.execute(query, (user_mobile, contact_mobile))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting contact: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def delete_contact_messages(self, user_mobile, contact_mobile):
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()
            query = """
                DELETE FROM messages_sign_chat
                WHERE (sender_mobile = %s AND receiver_mobile = %s)
                   OR (sender_mobile = %s AND receiver_mobile = %s)
            """
            cursor.execute(query, (user_mobile, contact_mobile, contact_mobile, user_mobile))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting messages: {e}")
            return False
        finally:
            if conn:
                conn.close()