import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

export async function POST(request: Request) {
  try {
    const { email, otp, name } = await request.json();

    // Create a transporter with bloodvision.org@gmail.com account
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: 'bloodvision.org@gmail.com',
        pass: 'fiuu dpvu mryv mhvq', // Your app password
      },
    });

    // Email template
    const mailOptions = {
      from: '"Blood Vision" <bloodvision.org@gmail.com>',
      to: email,
      subject: 'Blood Vision - Your Verification Code',
      html: `
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
          <!-- Header with Logo -->
          <div style="background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); padding: 20px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">Blood Vision</h1>
          </div>
          
          <!-- Main Content -->
          <div style="background-color: #ffffff; padding: 30px; color: #333333;">
            <p style="font-size: 16px;">Hello ${name || 'there'},</p>
            <p style="font-size: 16px;">Thank you for registering with Blood Vision. To verify your account, please use the following verification code:</p>
            
            <!-- OTP Box -->
            <div style="background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; margin: 25px 0;">
              <p style="margin: 0 0 10px 0; color: #666666; font-size: 14px;">Your verification code is:</p>
              <div style="font-family: monospace; font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #d32f2f;">
                ${otp}
              </div>
              <p style="margin: 10px 0 0 0; color: #666666; font-size: 12px;">This code will expire in 5 minutes</p>
            </div>
            
            <p style="font-size: 16px;">If you didn't request this code, please ignore this email or contact support if you have concerns.</p>
            
            <p style="font-size: 16px; margin-top: 30px;">Best regards,<br>The Blood Vision Team</p>
          </div>
          
          <!-- Footer -->
          <div style="background-color: #f5f5f5; padding: 20px; text-align: center; font-size: 12px; color: #666666;">
            <p>© ${new Date().getFullYear()} Blood Vision. All rights reserved.</p>
            <p>This is an automated message, please do not reply to this email.</p>
          </div>
        </div>
      `
    };

    // Send the email
    await transporter.sendMail(mailOptions);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error sending email:', error);
    return NextResponse.json(
      { error: 'Failed to send verification email' },
      { status: 500 }
    );
  }
} 