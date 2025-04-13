import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

export async function POST(request: Request) {
  try {
    const { email, patientName, reportData, reportDate, bloodGroup } = await request.json();

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
      subject: 'Blood Vision - Your Blood Group Report',
      html: `
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
          <!-- Header with Logo -->
          <div style="background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); padding: 20px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">Blood Vision</h1>
          </div>
          
          <!-- Main Content -->
          <div style="background-color: #ffffff; padding: 30px; color: #333333;">
            <p style="font-size: 16px;">Hello ${patientName || 'there'},</p>
            <p style="font-size: 16px;">Please find your blood group test report as requested. Here are your results:</p>
            
            <!-- Report Box -->
            <div style="background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 25px 0;">
              <div style="text-align: center; margin-bottom: 15px;">
                <p style="margin: 0 0 10px 0; color: #666666; font-size: 14px;">Your Blood Group is:</p>
                <div style="font-family: monospace; font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #d32f2f;">
                  ${bloodGroup}
                </div>
              </div>
              
              <div style="border-top: 1px solid #e0e0e0; padding-top: 15px; margin-top: 15px;">
                <p style="margin: 0; font-size: 14px;"><strong>Patient Name:</strong> ${patientName}</p>
                <p style="margin: 5px 0; font-size: 14px;"><strong>Report Date:</strong> ${reportDate}</p>
                <p style="margin: 5px 0; font-size: 14px;"><strong>Patient ID:</strong> ${reportData.id || 'N/A'}</p>
              </div>
            </div>
            
            <p style="font-size: 16px;">The complete report is attached to this email.</p>
            <p style="font-size: 16px; margin-top: 10px;">If you have any questions about these results, please consult with your healthcare provider.</p>
            
            <p style="font-size: 16px; margin-top: 30px;">Best regards,<br>The Blood Vision Team</p>
          </div>
          
          <!-- Footer -->
          <div style="background-color: #f5f5f5; padding: 20px; text-align: center; font-size: 12px; color: #666666;">
            <p>© ${new Date().getFullYear()} Blood Vision. All rights reserved.</p>
            <p>This is an automated message, please do not reply to this email.</p>
          </div>
        </div>
      `,
      attachments: [
        {
          filename: `${reportData.id || 'blood-report'}_Report.pdf`,
          content: reportData.pdfBase64.split(';base64,').pop(),
          encoding: 'base64',
        }
      ]
    };

    // Send the email
    await transporter.sendMail(mailOptions);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error sending report email:', error);
    return NextResponse.json(
      { error: 'Failed to send report email' },
      { status: 500 }
    );
  }
} 